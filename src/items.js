/** Read the items a `system_one` call judges: records from a file, inline texts, or images. */
import { readFile } from 'node:fs/promises'
import { extname, relative } from 'node:path'

const IMAGE_TYPES = { '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp' }

/**
 * Minimal RFC 4180 CSV parser: quoted fields, escaped quotes, newlines inside quotes. Nothing in the file is lost: a
 * record of empty fields (`,`) is a record, a header written twice names its second column with the next free name
 * (`text_2`), and the cells past the header's last are read under their column's number (`column_3`). `notes`, when
 * given, says how a header was read.
 */
export function parseCsv(text, notes) {
  const rows = []
  let row = []
  let field = ''
  let quoted = false
  // a line with nothing on it is no record; one with a comma or quotes is
  let written = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (c === '"') {
        quoted = false
      } else {
        field += c
      }
    } else if (c === '"') {
      quoted = true
      written = true
    } else if (c === ',') {
      row.push(field)
      field = ''
      written = true
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field)
      field = ''
      if (written || row.some(cell => cell !== '')) rows.push(row)
      row = []
      written = false
    } else {
      field += c
    }
  }
  row.push(field)
  if (written || row.some(cell => cell !== '')) rows.push(row)
  if (rows.length === 0) return []
  const [header, ...body] = rows
  const names = []
  const taken = new Set()
  const named = name => {
    taken.add(name)
    names.push(name)
  }
  for (const [index, cell] of header.entries()) {
    const name = cell.trim()
    if (!taken.has(name)) {
      named(name)
      continue
    }
    let other = `${name}_2`
    for (let n = 3; taken.has(other); n++) other = `${name}_${n}`
    notes?.push(`the CSV header names "${name}" twice: its column ${index + 1} is read as "${other}"`)
    named(other)
  }
  const widest = Math.max(header.length, ...body.map(cells => cells.length))
  for (let index = header.length; index < widest; index++) {
    let name = `column_${index + 1}`
    while (taken.has(name)) name += '_'
    named(name)
  }
  if (widest > header.length) {
    const longer = body.filter(cells => cells.length > header.length).length
    notes?.push(`${longer} CSV row(s) have more cells than the header names: the cells past it are read as ${names.slice(header.length).map(name => `"${name}"`).join(', ')}`)
  }
  return body.map(cells => Object.fromEntries(names.map((name, index) => [name, cells[index] ?? ''])))
}

/**
 * A value's JSON text, as JSON.stringify writes it, and where each part of it sits (`parts`: each key, string, number
 * and literal, with its path from the value; a string's or key's range is inside its quotes), so that words marked in
 * the text can be traced to the part that holds them. `parts` is undefined for a value JSON.stringify writes some
 * other way (a toJSON method, a number JSON has none for).
 */
export function jsonText(value) {
  const parts = []
  let text = ''
  const walk = (node, path) => {
    if (Array.isArray(node)) {
      text += '['
      node.forEach((child, index) => {
        if (index) text += ','
        walk(child, [...path, index])
      })
      text += ']'
      return
    }
    if (node && typeof node === 'object') {
      text += '{'
      Object.entries(node).forEach(([key, child], index) => {
        if (index) text += ','
        const quoted = JSON.stringify(key)
        parts.push({ path, key, start: text.length + 1, end: text.length + quoted.length - 1 })
        text += `${quoted}:`
        walk(child, [...path, key])
      })
      text += '}'
      return
    }
    const literal = JSON.stringify(node) ?? 'null'
    const quoted = typeof node === 'string' ? 1 : 0
    parts.push({ path, start: text.length + quoted, end: text.length + literal.length - quoted })
    text += literal
  }
  walk(value, [])
  const written = JSON.stringify(value) ?? ''
  return written === text ? { text, parts } : { text: written, parts: undefined }
}

/**
 * What System One reads of a record, and where each field sits in it (`segments`, UTF-16 ranges; `keyStart` where the
 * field's name starts; `json` and `parts` for a field that is not a string and reads as its JSON), so that the words
 * a span question marks can be traced back to their field, or to the field's name.
 */
function recordText(record, textField) {
  if (typeof record === 'string') return { text: record, segments: [{ start: 0, end: record.length }] }
  // a number, a boolean or null reads as its JSON, as a field holding one does
  if (record === null || typeof record !== 'object') {
    const { text, parts } = jsonText(record)
    return { text, segments: [{ start: 0, end: text.length, json: true, parts }] }
  }
  const shown = value => {
    if (typeof value === 'string') return { value }
    const { text, parts } = jsonText(value)
    return { value: text, json: true, parts }
  }
  let keys = Object.keys(record)
  if (textField) {
    // one field, or several as "subject,body"; a field named in another case is that field
    const named = textField.split(/\s*[,+|]\s*/).filter(Boolean).map(name => (record[name] !== undefined ? name : keys.find(key => key.toLowerCase() === name.toLowerCase()) ?? name))
    const missing = named.find(name => record[name] === undefined)
    if (missing !== undefined) throw new Error(`text_field "${missing}" is not a field of the items; they have ${keys.map(key => `"${key}"`).join(', ')}`)
    if (named.length === 1) {
      const { value, json, parts } = shown(record[named[0]])
      return { text: value, segments: [{ field: named[0], start: 0, end: value.length, ...(json ? { json, parts } : {}) }] }
    }
    keys = named
  }
  let text = ''
  const segments = []
  for (const [key, field] of keys.map(key => [key, record[key]])) {
    if (text) text += '\n'
    const keyStart = text.length
    text += `${key}: `
    const { value, json, parts } = shown(field)
    segments.push({ field: key, keyStart, start: text.length, end: text.length + value.length, ...(json ? { json, parts } : {}) })
    text += value
  }
  return { text, segments }
}

// an item's id, and the field it came from; an id the item does not carry is its number
function recordId(record, idField, index) {
  if (record !== null && typeof record === 'object') {
    if (idField && record[idField] !== undefined) return { id: String(record[idField]), idField }
    if (record.id !== undefined) return { id: String(record.id), idField: 'id' }
  }
  return { id: String(index + 1), numbered: true }
}

/** The format of a file of items, from its extension. */
export function formatOf(file) {
  const ext = extname(file).toLowerCase()
  return ext === '.jsonl' || ext === '.ndjson' ? 'jsonl' : ext === '.json' ? 'json' : ext === '.csv' ? 'csv' : 'lines'
}

// a file's name, after the last separator of either kind: Windows' paths use both
const fileName = path => path.split(/[\\/]/).pop()

/**
 * @param {object} options - `cwd` is the directory image sources are shown relative to; `notes`, when given, says how
 *   the items were read where that differs from the file (a CSV header, an id another item has)
 * @returns {Promise<Array<{id: string, idField?: string, text: string, images?: string[], source?: string, path?: string, record?: object | string,
 *   segments?: Array<{field?: string, keyStart?: number, start: number, end: number, json?: boolean, parts?: object[]}>, key?: string, line?: number, cut?: boolean}>>}
 *   `record` is what the item was read from, `idField` the field its id came from, `key` its key in a JSON object,
 *   `line` its line in a file of lines, and `cut` says that its text was longer than `maxChars`
 */
export async function loadItems({ file, items, images, textField, idField, maxItems, maxChars, cwd = process.cwd(), notes }) {
  const out = []
  if (images?.length) {
    for (const [index, path] of images.entries()) {
      const type = IMAGE_TYPES[extname(path).toLowerCase()]
      if (!type) throw new Error(`unsupported image type: ${path} (use PNG, JPEG or WebP)`)
      const data = await readFile(path)
      out.push({ id: fileName(path) || String(index + 1), numbered: true, text: '', images: [`data:${type};base64,${data.toString('base64')}`], source: relative(cwd, path) || path, path })
    }
  }
  if (file) {
    const raw = await readFile(file, 'utf8')
    const format = formatOf(file)
    let records
    let keys
    let lines
    if (format === 'jsonl') {
      records = raw.split('\n').filter(line => line.trim()).map((line, index) => {
        try {
          return JSON.parse(line)
        } catch {
          throw new Error(`${file}:${index + 1} is not valid JSON`)
        }
      })
    } else if (format === 'json') {
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed)) keys = Object.keys(parsed)
      records = Array.isArray(parsed) ? parsed : Object.values(parsed)
    } else if (format === 'csv') {
      records = parseCsv(raw, notes)
    } else {
      // each line as it is written, its indentation included; only its line ending goes
      lines = []
      records = []
      raw.split('\n').forEach((line, at) => {
        if (!line.trim()) return
        lines.push(at)
        records.push(line.replace(/\r$/, ''))
      })
    }
    records.forEach((record, index) => {
      const { text, segments } = recordText(record, textField)
      out.push({ ...recordId(record, idField, index), text, record, segments, ...(keys ? { key: keys[index] } : {}), ...(lines ? { line: lines[index] } : {}) })
    })
  }
  for (const [index, text] of (items ?? []).entries()) out.push({ id: String(index + 1), numbered: true, text, record: text, segments: [{ start: 0, end: text.length }] })
  if (out.length === 0) throw new Error('nothing to judge: pass file, items, or images')
  if (out.length > maxItems) throw new Error(`${out.length} items is more than the limit of ${maxItems}`)
  // Each id names one item. One the plugin gave (a row's or an inline item's number, an image's name) that an earlier
  // item has, as the first inline item's 1 beside a file's first row, becomes the next free number, or the image's
  // path; an id the items carry themselves stays as it is.
  const taken = new Set()
  const moved = []
  let next = 1
  for (const item of out) {
    if (item.numbered && taken.has(item.id)) {
      const was = item.id
      if (item.path && !taken.has(item.source)) item.id = item.source
      else {
        while (taken.has(String(next))) next++
        item.id = String(next)
      }
      moved.push(`${was} → ${item.id}`)
    }
    delete item.numbered
    taken.add(item.id)
  }
  if (moved.length) notes?.push(`${moved.length} item id(s) an earlier item had were given new ones: ${moved.slice(0, 6).join(', ')}${moved.length > 6 ? ', …' : ''}`)
  for (const item of out) {
    if (item.text.length > maxChars) {
      item.text = `${item.text.slice(0, maxChars)}…`
      item.cut = true
    }
  }
  return out
}
