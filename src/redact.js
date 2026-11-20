/** Redaction: the words span questions marked in each item, replaced by their label, in a copy of the items. */
import { readFile, writeFile } from 'node:fs/promises'
import { maskLabel, replaceSpans } from './spans.js'

// a number as a spreadsheet reads one, which a leading minus or plus does not turn into a formula
const PLAIN_NUMBER = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/

/**
 * One CSV cell: quoted when it holds a comma, a quote or a line break, and, when it starts as a spreadsheet formula
 * does (`=`, `+`, `-`, `@`, a tab or a carriage return), led by a `'` so that a spreadsheet shows it as text: a
 * record's id of `=1+1` is data, not a formula. A plain number stays as it is.
 */
export function csvCell(value) {
  let text = value === undefined || value === null ? '' : typeof value === 'string' ? value : JSON.stringify(value)
  if (/^[=+\-@\t\r]/.test(text) && !PLAIN_NUMBER.test(text)) text = `'${text}`
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
}

// a field of a copy under the name it has, whatever that is: `__proto__` too is a field of its own
const put = (object, key, value) => Object.defineProperty(object, key, { value, enumerable: true, writable: true, configurable: true })

/** An object of `entries`; a name that two of them take (after a key was renamed) keeps them apart with a number. */
function named(entries) {
  const out = {}
  for (const [key, value] of entries) {
    let name = key
    for (let n = 2; Object.hasOwn(out, name); n++) name = `${key} ${n}`
    put(out, name, value)
  }
  return out
}

/** Where each character of `text` starts in its JSON form (`at[i]`, inside the quotes), and where it ends (`at[length]`). */
function escapedAt(text) {
  const at = [0]
  for (let index = 0; index < text.length; index++) {
    if (/[\uD800-\uDBFF]/.test(text[index]) && /[\uDC00-\uDFFF]/.test(text[index + 1] ?? '')) {
      at.push(at[index] + 1, at[index] + 2)
      index++
      continue
    }
    at.push(at[index] + JSON.stringify(text[index]).length - 2)
  }
  return at
}

/** A string of a JSON text with the cuts that fall in it, mapped from its escaped form (`part`) onto its characters. */
function maskString(value, part, cuts) {
  const at = escapedAt(value)
  const first = offset => {
    let index = 0
    while (index < value.length && at[index + 1] <= offset) index++
    return index
  }
  const after = offset => {
    let index = value.length
    while (index > 0 && at[index - 1] >= offset) index--
    return index
  }
  const placed = cuts.map(cut => ({ ...cut, start: first(Math.max(cut.start, part.start) - part.start), end: after(Math.min(cut.end, part.end) - part.start) }))
  let out = ''
  let cursor = 0
  for (const cut of placed.sort((a, b) => a.start - b.start)) {
    if (cut.start < cursor || cut.end <= cut.start) continue
    out += value.slice(cursor, cut.start) + maskLabel(cut)
    cursor = cut.end
  }
  return out + value.slice(cursor)
}

const placeOf = (path, key) => JSON.stringify(key === undefined ? [path] : [path, key])

/**
 * A field's value that is not a string, with the cuts that fall in its JSON text (`cuts`, offsets into that text)
 * applied where they fall (`parts`, items.js `jsonText`): a string keeps its type with the marked words replaced, a
 * number or another literal becomes its `[LABEL]`, a key keeps its value under the name `rename` gives it, and arrays
 * stay arrays.
 */
function maskValue(value, parts, cuts, rename) {
  const hits = new Map()
  for (const part of parts) {
    const inside = cuts.filter(cut => cut.start < part.end && cut.end > part.start)
    if (inside.length) hits.set(placeOf(part.path, part.key), { part, cuts: inside })
  }
  const walk = (node, path) => {
    if (Array.isArray(node)) return node.map((child, index) => walk(child, [...path, index]))
    if (node && typeof node === 'object') {
      return named(Object.entries(node).map(([key, child]) => {
        const hit = hits.get(placeOf(path, key))
        return [hit ? rename(key, hit.cuts[0].label) : key, walk(child, [...path, key])]
      }))
    }
    const hit = hits.get(placeOf(path))
    if (!hit) return node
    return typeof node === 'string' ? maskString(node, hit.part, hit.cuts) : maskLabel(hit.cuts[0])
  }
  return walk(value, [])
}

// a part of a person's name that names them alone, as "Priya" of "Priya Raman"
const NAME_PART = /^\p{Lu}[\p{L}'’-]{2,}$/u
const WORD = /[\p{L}\p{N}]/u
// scripts written without spaces between words, where a letter beside a match is no sign of a longer word
const UNSPACED = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u

// Whether two characters side by side belong to one word, so that a match ending or starting between them is part of
// a longer word or number ("Ann" in "Annual", "4242" in "1424242").
const oneWord = (a, b) => WORD.test(a) && WORD.test(b) && !UNSPACED.test(a) && !UNSPACED.test(b)

// A marked string is masked wherever else it occurs in the item, as the model may mark a name only once; so is each
// part of a marked person's name ("Thanks Priya" after "Priya Raman"). Each recurrence is a whole word or number:
// one that a letter or a digit continues on either side is another word.
export function everywhere(text, spans) {
  const out = [...spans]
  for (const span of spans) {
    const value = text.slice(span.start, span.end)
    const parts = span.label === 'PERSON' ? value.split(/\s+/).filter(part => NAME_PART.test(part) && part !== value.trim()) : []
    for (const needle of [value, ...parts]) {
      if (needle.trim().length < 3) continue
      for (let at = text.indexOf(needle); at >= 0; at = text.indexOf(needle, at + 1)) {
        const end = at + needle.length
        if (oneWord(text[at - 1] ?? '', needle[0]) || oneWord(needle.at(-1), text[end] ?? '')) continue
        out.push({ ...span, start: at, end })
      }
    }
  }
  return out
}

// overlapping spans become one, under the label of the first
export function joined(spans) {
  const out = []
  for (const span of [...spans].sort((a, b) => a.start - b.start || b.end - a.end)) {
    const last = out.at(-1)
    if (last && span.start < last.end) last.end = Math.max(last.end, span.end)
    else out.push({ ...span })
  }
  return out
}

/**
 * An item with `spans` (UTF-16 ranges of its text) replaced by `[LABEL]`: its text, and a copy of its record with each
 * field's part of the spans replaced (items.js `segments`). A field that is not a string is masked inside its value,
 * which keeps its shape: a string in it keeps its type, a number or another literal becomes its `[LABEL]`, and an array
 * stays an array, as an array record does. A name that holds marked words, a field's own or a key inside its value,
 * takes the name `rename(name, label)` gives it (its `[LABEL]` by default). The item's id stays, so the copy's rows
 * still match the results. `replaced` counts the places masked: each span, and each other place where its words recur.
 */
export function redactItem(item, spans, { rename = (_name, label) => maskLabel({ label }) } = {}) {
  const id = item.segments?.find(segment => segment.field !== undefined && segment.field === item.idField)
  const cuts = joined(everywhere(item.text, spans).filter(span => !id || span.end <= id.start || span.start >= id.end))
  const text = replaceSpans(item.text, cuts, maskLabel)
  if (typeof item.record !== 'object' || item.record === null) return { text, record: text, replaced: cuts.length }
  const array = Array.isArray(item.record)
  const record = array ? [...item.record] : { ...item.record }
  const names = new Map()
  for (const segment of item.segments ?? []) {
    if (segment.field === undefined) continue
    // the field's own name, as "alice@example.test: hello" reads it
    if (!array && segment.keyStart !== undefined) {
      const inName = cuts.find(span => span.start < segment.keyStart + segment.field.length && span.end > segment.keyStart)
      if (inName) names.set(segment.field, rename(segment.field, inName.label))
    }
    const inside = cuts
      .filter(span => span.start < segment.end && span.end > segment.start)
      .map(span => ({ ...span, start: Math.max(span.start, segment.start) - segment.start, end: Math.min(span.end, segment.end) - segment.start }))
    if (inside.length === 0) continue
    if (!segment.json) {
      put(record, segment.field, replaceSpans(record[segment.field], inside, maskLabel))
      continue
    }
    if (segment.parts) {
      put(record, segment.field, maskValue(record[segment.field], segment.parts, inside, rename))
      continue
    }
    // a value whose JSON its parts do not describe is masked in that JSON
    const masked = replaceSpans(JSON.stringify(record[segment.field]), inside, maskLabel)
    try {
      put(record, segment.field, JSON.parse(masked))
    } catch {
      put(record, segment.field, masked)
    }
  }
  return { text, record: names.size ? named(Object.entries(record).map(([key, value]) => [names.get(key) ?? key, value])) : record, replaced: cuts.length }
}

/**
 * The redacted copy of a file of items, in the file's format; `records` holds each item's redacted record, and `keys`,
 * when given, the key each record of a JSON object takes in the copy. A file of lines keeps its other lines as they were.
 */
export async function writeRedacted({ file, format, output, items, records, keys }) {
  let body
  if (format === 'jsonl') body = `${records.map(record => JSON.stringify(record)).join('\n')}\n`
  else if (format === 'json') {
    const keyed = items.length > 0 && items.every(item => item.key !== undefined)
    body = `${JSON.stringify(keyed ? Object.fromEntries(items.map((item, index) => [keys?.[index] ?? item.key, records[index]])) : records, null, 2)}\n`
  } else if (format === 'csv') {
    const columns = [...new Set(records.flatMap(record => Object.keys(record)))]
    body = `${[columns.map(csvCell).join(','), ...records.map(record => columns.map(column => csvCell(record[column])).join(','))].join('\n')}\n`
  } else if (file) {
    const lines = (await readFile(file, 'utf8')).split('\n')
    items.forEach((item, index) => {
      if (item.line !== undefined) lines[item.line] = lines[item.line].replace(item.record, () => records[index])
    })
    body = lines.join('\n')
  } else {
    body = `${records.join('\n')}\n`
  }
  await writeFile(output, body)
}
