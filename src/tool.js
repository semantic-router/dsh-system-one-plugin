/** The model-facing `system_one` tool: one rubric, many items, calibrated answers. */
import { randomBytes } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { basename, dirname, extname, isAbsolute, join, relative, resolve } from 'node:path'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { QUESTIONS_PER_REQUEST, REQUEST_TOKENS, compactAnswer, requestSlices, stateCost, unavailable, unavailableText } from './client.js'
import { formatOf, loadItems } from './items.js'
import { inputAt, kindAt, mustBeFile, noFile, sameFile } from './paths.js'
import { deniedText, mayWrite } from './permissions.js'
import { annotatePng, undrawable } from './png.js'
import { personalShape, pseudonym } from './pseudonyms.js'
import { ROUTER_SIGNALS, SENSITIVE_LABELS } from './questions.js'
import { csvCell, everywhere, joined, redactItem, writeRedacted } from './redact.js'
import { kindOf, planQuestions, put, readQuestions, splitArguments, unanswerable } from './rubric.js'
import { apart, excerpts, labelCounts, maskAround, notPersonal, placeSpans, trimSpan } from './spans.js'
import { clip } from './trace.js'

export { parseQuestions } from './rubric.js'

// what the sources of one call may hold
const SOURCE_CHARS = 24000
// a box answer at least this confident found its target
export const FOUND_AT = 0.4
// a word that is more likely unsupported than not, and personal data as the privacy reflex masks it (privacyAt); a
// question's threshold says otherwise
const HALU_AT = 0.5
const PII_AT = 0.5
const IMAGES_SHOWN = 12
// a batch waits out a spent per-minute budget (the key's, or its network address's) as the gateway's Retry-After
// says, rather than fail; the reflexes keep their short limits
const BATCH_LIMITS = { retries: 6, waitMs: 60000 }
// the signals that look for jailbreaks, harm and refusals: the items they read are often what System Two should not
// read, so their results keep ids and answers only
const SAFETY_SIGNALS = new Set(['attack', 'p_harm', 'r_harm', 'r_refusal'])

// one question of each kind, as the description and the questions parameter show them
export const EXAMPLE_QUESTIONS = {
  team: { type: 'choice', instructions: 'Which team should handle this ticket?', criteria: { billing: 'payments, invoices, refunds', engineering: 'bugs, errors, outages', sales: 'plans and pricing' } },
  angry: { type: 'noul', instructions: 'Is the customer angry?' },
  urgency: { type: 'score', instructions: 'How urgent is this ticket?', criteria: ['1 = can wait', '2 = this week', '3 = today'] },
  marks: { type: 'span', criteria: { ERROR: 'an error code or message the customer quotes', COMPETITOR: 'another product the customer names' }, threshold: 0.5 },
  personal: { preset: 'pii' },
  jailbreak: { signal: 'attack' },
}
const asJson = value => JSON.stringify(value)

const DESCRIPTION = `Ask System One, a fast decision model, typed questions about many items at once (the records of a JSONL, JSON or CSV file, the lines of a text file, a list of strings, or images) and get calibrated probabilities back in seconds, instead of reading the items yourself. Use it whenever the same judgment applies to many items: routing, labeling, filtering, triage, scoring, ranking, moderation, finding personal data, checking claims against sources, spotting problems in screenshots.

System One answers every question for every item, writes one result row per item to an output file (JSONL or CSV) and returns a summary. Read the output file afterwards only for the rows you need.

questions is one object: each key is an id you choose, each value one question. The kinds, one example each:
- choice: pick one option; criteria names 2 or more options, each with a short description. ${asJson({ team: EXAMPLE_QUESTIONS.team })}
- noul: yes or no, without criteria; instructions say what counts as yes; the answer is the probability of yes. ${asJson({ angry: EXAMPLE_QUESTIONS.angry })}
- score: an ordered scale; criteria lists 2 to 10 levels, lowest first; the answer is the expected level, from 0, or on the levels' own numbers when each starts with one. ${asJson({ urgency: EXAMPLE_QUESTIONS.urgency })}
- box (images only): where something is; instructions name it; the answer is a box in fractions of the image with a confidence (below 0.4: not found). ${asJson({ where: { type: 'box', instructions: 'the spike in the plotted line' } })}
- span: the words of each text that carry your labels (error codes, product names, requests, quotes), not one category per item; criteria names every label with what it marks; threshold (0 to 1, optional) drops less probable marks. ${asJson({ marks: EXAMPLE_QUESTIONS.marks })}
- preset: ${asJson({ personal: EXAMPLE_QUESTIONS.personal })} marks personal data (${SENSITIVE_LABELS.join(', ')}; "labels" picks others it knows, such as ORGANIZATION, DATE_TIME and GPE, or "all"); {"preset": "halu"} the words of each item its sources do not support; {"preset": "evidence"} the words of the sources that back each item. halu and evidence need sources.
- signal: the vLLM Semantic Router's trained questions. ${asJson({ jailbreak: EXAMPLE_QUESTIONS.jailbreak })} asks attack (jailbreak or benign); also p_harm (unsafe or safe request), domain (subject area), factcheck, feedback, modality, scenario; r_harm and r_refusal judge an answer to a user request (records with fields "user" and "answer", or state_fields).

Rules:
- Every field of a question goes inside that question, under its own id: type, instructions, criteria, threshold. Use each id once; nothing else goes beside the questions.
- A span question lists all its labels in criteria; its instructions, if any, only say what to look for. Describe each label precisely, and say in context what the items are and whose (e.g. "Support tickets sent to Lumen Cloud"), so a label such as COMPETITOR knows which product is yours.
- box questions read images; span questions, presets and signals read text.
- redact copies the items with what preset pii marked replaced by its [LABEL] (else what your span questions marked); annotate copies the images with the boxes found.
Span questions, presets and signals are answered by span_model; choice, noul, score and box questions by model.

For one screenshot you act on (where to click, whether an action worked), use system_one_screen.`

// the tool's parameters, for arguments that ran into questions (splitArguments)
const PARAMETERS = Object.fromEntries(['questions', 'file', 'text_field', 'id_field', 'items', 'images', 'context', 'sources', 'state_fields', 'redact', 'annotate', 'output', 'model', 'span_model'].map(name => [name, true]))

/**
 * Where a call's rows go by default: beside its file, or beside the one folder its images are in. Only the file's own
 * extension goes, after the last separator of either kind, so that `project.v1\tickets` keeps its folder on Windows.
 */
function defaultOutput(cwd, file, images) {
  if (file) return `${file.replace(/(?<=[^./\\])\.[^./\\]+$/, '')}.s1.jsonl`
  const folders = [...new Set((images ?? []).map(path => dirname(path)))]
  const within = folders.length === 1 ? relative(cwd, folders[0]) : ''
  return within && !within.startsWith('..') && !isAbsolute(within) ? `${folders[0]}.s1.jsonl` : 'system-one-results.jsonl'
}

/**
 * Where the annotated copy of one image goes: the file named, into the folder named, or beside the image; one that
 * would land on a file the call reads or writes (`kept`: its image, another image, a source, the results) goes beside
 * its own image as `<name>.annotated.png`.
 */
async function annotatedPath(annotateTo, item, kept) {
  const named = annotateTo.file ?? (annotateTo.beside ? item.path : join(annotateTo.dir, basename(item.path)))
  if (!(await inputAt(named, kept))) return named
  const beside = `${item.path.slice(0, item.path.length - extname(item.path).length)}.annotated.png`
  const taken = await inputAt(beside, kept)
  if (taken) throw new Error(`annotate would write the copy of ${item.source} over ${taken === item.path ? 'the image itself' : basename(taken)}: name a folder of its own for the annotated copies`)
  return beside
}

/**
 * The number of a score question's first level when its levels carry their own numbers, as ["1 = vague", "2 = clear",
 * "3 = concrete"]: its answers are then given on that numbering. 0 otherwise: the model's answer counts levels from 0.
 */
function levelBase(criteria) {
  const numbers = (Array.isArray(criteria) ? criteria : []).map(level => String(level).match(/^\s*(\d+)(?!\d|[.,]\d)/)?.[1])
  if (numbers.length < 2 || numbers.some(number => number === undefined)) return 0
  const first = Number(numbers[0])
  return numbers.every((number, index) => Number(number) === first + index) ? first : 0
}

/** How many of `values` there are of each, in the order they first come: a value of any name counts as its own. */
function tally(values) {
  const counts = new Map()
  for (const [value, n] of values) counts.set(value, (counts.get(value) ?? 0) + n)
  return counts
}

/**
 * The names a score's levels go by in the summary: the words before each one's colon (`Low` of "Low: can wait"), or
 * the whole level where two would share them, each name its own.
 */
function levelNames(criteria) {
  const short = criteria.map(level => String(level).split(':')[0])
  const names = new Set(short).size === short.length ? short : criteria.map(String)
  const seen = new Set()
  return names.map(name => {
    let own = name
    for (let n = 2; seen.has(own); n++) own = `${name} (${n})`
    seen.add(own)
    return own
  })
}

/**
 * The call's summary for System Two. `hidden` holds the span questions whose words it never shows (personal data, or
 * what a redacted copy replaces); `applies(id, index)` whether a question was asked of the item at `index` at all, as a
 * box question is not of a text.
 */
function summarize(questions, rows, asked, { hidden = new Set(), applies = () => true } = {}) {
  const summary = {}
  for (const [id, question] of Object.entries(questions)) {
    const asks = rows.filter((_, index) => applies(id, index))
    const answers = asks.map(row => ({ id: row.id, a: row.answers[id] })).filter(entry => entry.a && !entry.a.error)
    const failed = asks.length - answers.length
    // why they failed, once, so that System Two can fix the question
    const why = failed ? asks.map(row => row.answers[id]).find(a => a?.error) : undefined
    const error = why ? clip([why.error, why.reason].filter(value => typeof value === 'string' && value).join(': '), 200) : undefined
    const kind = asked[id].kind
    if (kind === 'span') {
      const labels = tally(answers.flatMap(({ a }) => Object.entries(a.labels)))
      const marked = answers.filter(({ a }) => a.n > 0)
      const top = [...marked].sort((x, y) => y.a.n - x.a.n || y.a.p - x.a.p)
      put(summary, id, {
        type: 'span',
        ...(question.preset ? { preset: question.preset } : {}),
        labels: Object.fromEntries([...labels].sort((x, y) => y[1] - x[1])),
        items: marked.length,
        of: answers.length,
        top: top.slice(0, 8).map(({ id: itemId, a }) => [itemId, a.n]),
        // personal data, and what a redacted copy replaces, stay in the results file, not in what System Two reads
        ...(hidden.has(id) ? {} : {
          examples: top.slice(0, 6).map(({ id: itemId, a }) => {
            const best = a.spans.reduce((x, y) => (y.p > x.p ? y : x))
            return [itemId, best.label, clip(best.text.replace(/\s+/g, ' ').trim(), 120), best.p, ...(best.source ? [placeOf(best)] : [])]
          }),
        }),
        failed,
        ...(error ? { error } : {}),
      })
    } else if (question.type === 'choice' || question.signal) {
      const counts = Object.fromEntries(tally(answers.map(({ a }) => [a.choice, 1])))
      const unsure = answers.filter(({ a }) => a.confidence < 0.6).map(({ id: itemId }) => itemId)
      put(summary, id, { type: 'choice', ...(question.signal ? { signal: question.signal } : {}), counts, unsure: unsure.slice(0, 12), unsureCount: unsure.length, failed, ...(error ? { error } : {}) })
    } else if (question.type === 'noul') {
      const sorted = [...answers].sort((x, y) => y.a.yes - x.a.yes)
      const yesCount = answers.filter(({ a }) => a.yes >= 0.5).length
      put(summary, id, { type: 'noul', yes: yesCount, no: answers.length - yesCount, top: sorted.slice(0, 8).map(({ id: itemId, a }) => [itemId, a.yes]), failed, ...(error ? { error } : {}) })
    } else if (question.type === 'score') {
      const levels = question.criteria.map(() => 0)
      const base = levelBase(question.criteria)
      let total = 0
      for (const { a } of answers) {
        total += a.score
        levels[Math.min(levels.length - 1, Math.max(0, Math.round(a.score - base)))]++
      }
      const sorted = [...answers].sort((x, y) => y.a.score - x.a.score)
      put(summary, id, {
        type: 'score',
        mean: answers.length ? Math.round((total / answers.length) * 100) / 100 : null,
        levels: Object.fromEntries(levelNames(question.criteria).map((name, index) => [name, levels[index]])),
        top: sorted.slice(0, 8).map(({ id: itemId, a }) => [itemId, a.score]),
        failed,
        ...(error ? { error } : {}),
      })
    } else {
      const seen = answers.filter(({ a }) => a.box && a.found)
      const corners = ({ left, top, right, bottom }) => [left, top, right, bottom].map(value => Math.round(value * 1000) / 1000)
      put(summary, id, { type: 'box', found: seen.length, of: answers.length, boxes: seen.slice(0, 12).map(({ id: itemId, a }) => [itemId, corners(a.box), a.confidence]), failed, ...(error ? { error } : {}) })
    }
  }
  return summary
}

// the columns every row of a CSV and of the first rows has, beside the answers
const ROW_COLUMNS = ['id', 'pseudonym']

/** The columns one question's answers take in a CSV row and in the first rows. */
function columnsOf(id, question, kind) {
  if (kind === 'span') return [id, `${id}_labels`, `${id}_spans`]
  if (question.type === 'choice' || question.signal) return [id, `${id}_confidence`]
  if (question.type === 'noul') return [`${id}_yes`]
  if (question.type === 'score') return [id]
  return [id, `${id}_confidence`]
}

/**
 * Why the questions' answers cannot share a row: a question named as a column every row has (`id`), or two questions
 * whose answers take one column (`team`'s `team_confidence` and a question named `team_confidence`). Undefined when
 * every answer has a column of its own.
 */
function columnClash(questions, asked) {
  const owners = new Map(ROW_COLUMNS.map(column => [column, undefined]))
  for (const [id, question] of Object.entries(questions)) {
    for (const column of columnsOf(id, question, asked[id].kind)) {
      if (!owners.has(column)) {
        owners.set(column, id)
        continue
      }
      const first = owners.get(column)
      return first === undefined
        ? `question id "${id}" names the ${column} column, which every result row has for its item: rename the question, as "${id}_answer"`
        : `questions "${first}" and "${id}" would both write the column ${column} of the results: rename one of them`
    }
  }
  return undefined
}

/**
 * One row as a CSV and the first rows have it: a column per answer. `hidden` holds the span questions whose words stay
 * out of it; `unasked` the questions not asked of this item, which take no cell.
 */
function flatRow(row, questions, asked, { hidden = new Set(), unasked = new Set() } = {}) {
  const flat = { id: row.id, ...(row.pseudonym ? { pseudonym: row.pseudonym } : {}) }
  for (const [id, question] of Object.entries(questions)) {
    if (unasked.has(id)) continue
    const a = row.answers[id]
    if (!a || a.error) {
      put(flat, id, 'error')
      continue
    }
    if (asked[id].kind === 'span') {
      put(flat, id, a.n)
      if (a.n) put(flat, `${id}_labels`, Object.entries(a.labels).map(([label, n]) => `${label} ${n}`).join('; '))
      if (a.n && !hidden.has(id)) put(flat, `${id}_spans`, a.spans.map(span => `${span.label}: ${span.text}`).join(' | '))
    } else if (question.type === 'choice' || question.signal) {
      put(flat, id, a.choice)
      put(flat, `${id}_confidence`, a.confidence)
    } else if (question.type === 'noul') {
      put(flat, `${id}_yes`, a.yes)
    } else if (question.type === 'score') {
      put(flat, id, a.score)
    } else {
      put(flat, id, JSON.stringify(a.box))
      put(flat, `${id}_confidence`, a.confidence)
    }
  }
  return flat
}

async function writeResults(path, rows, questions, asked, show = () => ({})) {
  if (extname(path).toLowerCase() === '.csv') {
    const flat = rows.map((row, index) => flatRow(row, questions, asked, show(index)))
    const columns = [...new Set(flat.flatMap(row => Object.keys(row)))]
    // a header is a cell like any other: a question id with a comma, a quote or a line break keeps its column
    const lines = [columns.map(csvCell).join(','), ...flat.map(row => columns.map(column => csvCell(row[column])).join(','))]
    await writeFile(path, lines.join('\n') + '\n')
  } else {
    await writeFile(path, rows.map(row => JSON.stringify({ id: row.id, ...(row.pseudonym ? { pseudonym: row.pseudonym } : {}), text: row.text, answers: row.answers })).join('\n') + '\n')
  }
}

const listed = (entries, max = 8) => entries.slice(0, max).join(', ') + (entries.length > max ? ', …' : '')

function renderSummary(value) {
  const models = value.spanModel ? `${value.model} and ${value.spanModel}` : value.model
  const lines = [
    `System One (${models}) answered ${value.questions} question(s) for ${value.items} item(s) in ${value.seconds} s (${value.requests} requests).${value.output ? ` Results: ${value.output}` : ''}`,
  ]
  // what the call was read as, where it was written another way, so a wrong reading can be put right
  if (value.notes?.length) lines.push(`Notes: ${value.notes.join('; ')}.`)
  for (const [id, s] of Object.entries(value.summary)) {
    if (s.type === 'span') {
      const counts = Object.entries(s.labels).map(([label, n]) => `${label} ${n}`)
      const items = `${s.items} of ${s.of} item(s)`
      if (s.preset === 'halu') lines.push(`- ${id}: words the sources do not support in ${items}${s.examples.length ? `: ${listed(s.examples.map(([itemId, , text]) => `${itemId} ("${text}")`), 6)}` : ''}`)
      else if (s.preset === 'evidence') lines.push(`- ${id}: backed by the sources: ${items}${s.examples.length ? `; e.g. ${listed(s.examples.map(([itemId, , text, , where]) => `${itemId}: "${text}"${where ? ` (${where})` : ''}`), 3)}` : ''}`)
      else if (s.preset === 'pii') lines.push(`- ${id}: personal data in ${items}${counts.length ? `: ${counts.join(', ')}; most in ${listed(s.top.map(([itemId, n]) => `${itemId} (${n})`))}` : ''}`)
      // what a redacted copy replaces is counted, as personal data is, and its words are not shown
      else if (!s.examples) lines.push(`- ${id}: ${counts.length ? `${counts.join(', ')} in ${items}; most in ${listed(s.top.map(([itemId, n]) => `${itemId} (${n})`))}` : `nothing marked in ${items}`}`)
      else lines.push(`- ${id}: ${counts.length ? counts.join(', ') : 'nothing marked'} in ${items}${s.examples.length ? `; e.g. ${listed(s.examples.map(([itemId, label, text]) => `${itemId} ${label} "${text}"`), 4)}` : ''}`)
    } else if (s.type === 'choice') {
      lines.push(`- ${id}${s.signal ? ` (signal ${s.signal})` : ''}: ${Object.entries(s.counts).sort((a, b) => b[1] - a[1]).map(([k, n]) => `${k} ${n}`).join(', ')}${s.unsureCount ? `; unsure (confidence < 0.6): ${s.unsureCount} [${s.unsure.join(', ')}]` : ''}`)
    } else if (s.type === 'noul') {
      lines.push(`- ${id}: yes ${s.yes}, no ${s.no}; highest: ${s.top.map(([itemId, p]) => `${itemId} (${p})`).join(', ')}`)
    } else if (s.type === 'score') {
      lines.push(`- ${id}: mean ${s.mean}; ${Object.entries(s.levels).map(([k, n]) => `${k} ${n}`).join(', ')}; highest: ${s.top.map(([itemId, v]) => `${itemId} (${v})`).join(', ')}`)
    } else {
      lines.push(`- ${id}: found in ${s.found} of ${s.of} item(s)${s.boxes.length ? ` (box as [left, top, right, bottom], fractions of the image): ${s.boxes.map(([itemId, box, c]) => `${itemId} [${box.join(', ')}] conf ${c}`).join('; ')}` : ''}`)
    }
    if (s.failed) lines.push(`  (${s.failed} item(s) failed for ${id}${s.error ? `: ${s.error}` : ''})`)
  }
  if (value.redacted) lines.push(`Redacted copy: ${value.redacted.output} (${value.redacted.spans} span(s) in ${value.redacted.items} item(s) replaced by their [LABEL])`)
  if (value.annotated) {
    const none = value.annotated.skipped.length ? 'none' : 'none: no box was found'
    lines.push(`Annotated copies (boxes in red): ${value.annotated.files.length ? value.annotated.files.join(', ') : none}${value.annotated.skipped.length ? `; not drawn on, their boxes are in the results: ${value.annotated.skipped.join(', ')}` : ''}`)
  }
  if (value.preview.length) {
    lines.push('First rows:')
    for (const row of value.preview) lines.push(`  ${JSON.stringify(row)}`)
  }
  return lines.join('\n')
}

/** The result's facts as the Web UI's system_one row draws them (web/client.js), kept on the call's result. */
export function presentation(value, ui) {
  if (ui === 'off') return { systemOne: 1, ui }
  const { model, items, questions, requests, seconds, output, summary } = value
  const extra = Object.fromEntries(['spanModel', 'examples', 'images', 'redacted', 'annotated', 'sources', 'pseudonyms', 'partial'].filter(key => value[key] !== undefined).map(key => [key, value[key]]))
  return { systemOne: 1, ui, model, items, questions, requests, seconds, output, summary, ...extra }
}

/** The parts of each item's state from `state_fields`: {"part": "field"}, or a list of fields that keep their names. */
function fieldsOf(value) {
  if (value === undefined || value === null || value === '') return undefined
  let parsed = value
  if (typeof value === 'string') {
    try {
      parsed = JSON.parse(value)
    } catch {
      parsed = value.split(',').map(name => name.trim()).filter(Boolean)
    }
  }
  const map = Array.isArray(parsed) ? Object.fromEntries(parsed.map(name => [String(name), String(name)]))
    : parsed && typeof parsed === 'object' ? Object.fromEntries(Object.entries(parsed).map(([part, field]) => [part, String(field)])) : undefined
  if (!map || Object.keys(map).length === 0) throw new Error('state_fields names the parts of each item: {"part": "field"}, or a list of fields')
  return map
}

/**
 * The sources' text, as one context with a `[path]` line before each file, and where each file starts in it, and how
 * many lines its text starts below the file's first (`skipped`: the blank lines the context leaves out).
 */
async function readSources(paths, cwd, before) {
  const blocks = []
  let context = before ? `${before}\n\n` : ''
  for (const [index, path] of paths.entries()) {
    const full = resolve(cwd, path)
    const raw = await readFile(full, 'utf8')
    const text = raw.trim()
    const skipped = raw.slice(0, raw.length - raw.trimStart().length).match(/\n/g)?.length ?? 0
    const shown = relative(cwd, full) || full
    if (index > 0) context += '\n\n'
    context += `[${shown}]\n`
    blocks.push({ source: shown, start: context.length, text, skipped })
    context += text
  }
  if (context.length > SOURCE_CHARS) throw new Error(`the sources hold ${context.length} characters, and System One reads at most ${SOURCE_CHARS}: pass fewer or shorter files`)
  return { context, blocks }
}

// the file and line an offset of the sources' context falls on, counted in the file as it is
function sourceAt(blocks, index) {
  const block = [...blocks].reverse().find(entry => entry.start <= index)
  if (!block) return undefined
  return { source: block.source, line: 1 + block.skipped + (block.text.slice(0, index - block.start).match(/\n/g)?.length ?? 0) }
}

/** Where an evidence span is: a source's file and line, or the record's own field that it read before the sources. */
const placeOf = span => (span.line ? `${span.source}:${span.line}` : span.source)

const round3 = value => Math.round(value * 1000) / 1000

// the pseudonym key when none is given: one for the process
let ownKey
const processKey = async () => (ownKey ??= randomBytes(32))

/**
 * @param {object} deps
 * @param {ReturnType<import('./client.js').createDecisionClient>} deps.client
 * @param {(entry: object) => void} deps.trace
 * @param {object} deps.config
 * @param {(exec: object) => Promise<string>} [deps.workspaceOf] - the calling session's working directory, which
 *   relative paths resolve against
 * @param {(exec: object) => Promise<object | undefined>} [deps.writePolicyOf] - where the calling session lets tools
 *   write (permissions.js `writePolicy`); undefined writes anywhere
 * @param {() => Promise<Buffer>} [deps.pseudonymKey] - the key of the pseudonyms of item ids that are personal data
 *   (pseudonyms.js); by default one that lasts as long as the process
 */
export function systemOneTool({ client, trace, config, workspaceOf = async () => process.cwd(), writePolicyOf = async () => undefined, pseudonymKey = processKey }) {
  return defineTool({
    name: config.toolName,
    description: DESCRIPTION,
    parameters: {
      questions: {
        type: 'json',
        required: true,
        description: 'One object: question id -> question ({"type": "choice" | "noul" | "score" | "box" | "span", "instructions": ..., "criteria": ...}, {"preset": "pii" | "halu" | "evidence"} or {"signal": ...}), as the description shows. Each question\'s fields go inside it.',
        examples: [EXAMPLE_QUESTIONS],
      },
      file: { type: 'string', description: 'Path to a .jsonl, .json (array), .csv, or .txt (one item per line) file holding the items.' },
      text_field: { type: 'string', description: 'For .jsonl/.json/.csv: the field holding each item\'s text, or several, as "subject,body". Default: all fields.' },
      id_field: { type: 'string', description: 'For .jsonl/.json/.csv: the field holding each item\'s id. Default: "id" when present, else the row number.' },
      items: { type: 'array', items: { type: 'string' }, description: 'Inline items, for short lists without a file.' },
      images: { type: 'array', items: { type: 'string' }, description: 'Image paths (PNG, JPEG, WebP); each image is one item.' },
      context: { type: 'string', description: 'Text placed before every item: what the items are and whose (e.g. "Support tickets sent to Lumen Cloud, a data product"). It helps every question.' },
      sources: { type: 'array', items: { type: 'string' }, description: 'Files the items should agree with (docs, a policy, a changelog). Each item is checked against them: preset halu marks what they do not support, preset evidence the words of the sources that back the item.' },
      state_fields: {
        type: 'json',
        description: 'For records: the parts System One reads of each item, as {"part": "field"}: "context" and "answer" for halu and evidence (with sources, a record\'s "context", as the message a reply answers, is read before them), "user" and "answer" for the signals r_harm and r_refusal.',
        examples: [{ user: 'question', answer: 'reply' }],
      },
      redact: {
        oneOf: [{ type: 'boolean' }, { type: 'string' }, { type: 'null' }],
        description: 'Write a copy of the items with what preset pii marked (else what the span questions marked) replaced by its [LABEL], in the input\'s format: true for <file>.redacted.<ext>, or a path. Without a pii or span question, personal data is what it replaces.',
      },
      annotate: {
        oneOf: [{ type: 'boolean' }, { type: 'string' }, { type: 'null' }],
        description: 'For images with box questions: a directory for copies of the images with the boxes found drawn in red (PNG), or true for a copy beside each image (<name>.annotated.png).',
      },
      output: { type: 'string', description: 'Where to write one result row per item (.jsonl or .csv), or a folder for them. Default: <file>.s1.jsonl beside the input file, <folder>.s1.jsonl beside the images\' folder, or ./system-one-results.jsonl.' },
      model: { type: 'string', description: `Decision model for choice, noul, score and box questions. Default ${config.model}.` },
      span_model: { type: 'string', description: `Model for span questions, presets and signals. Default ${config.spanModel}.` },
    },
    output: {
      schema: { type: 'object', additionalProperties: true },
      render: (_args, value) => [{ type: 'text', text: renderSummary(value) }],
      presentationMeta: (_args, value) => presentation(value, config.systemOneUi),
    },
    async execute(args, exec) {
      const notes = []
      // the arguments System Two's tool-call markup ran into the questions, read as the arguments they are
      const split = splitArguments(args.questions)
      if (split) {
        const known = Object.keys(split.leaked).filter(key => Object.hasOwn(PARAMETERS, key) && key !== 'questions' && args[key] === undefined)
        args = { ...args, questions: split.head, ...Object.fromEntries(known.map(key => [key, split.leaked[key]])) }
        notes.push(`questions ran into the next arguments${known.length ? `: ${known.join(', ')} read from them` : ''}`)
      }
      const read = readQuestions(args.questions)
      let questions = read.questions
      notes.push(...read.notes)
      const cwd = await workspaceOf(exec)
      // A file goes only where DeepSeek Harness lets its own write tool write in this session: a read-only session keeps
      // the answers in this result, and a path outside the workspace is refused before anything is asked.
      const policy = await writePolicyOf(exec)
      const readOnly = policy?.roots.length === 0
      if (readOnly) notes.push(`no file written: ${policy.unknown ? 'this session\'s file policy could not be read' : 'this session is read-only'}, so the answers are in this result only`)
      const pathShown = path => {
        const inside = relative(cwd, path)
        return inside && !inside.startsWith('..') && !isAbsolute(inside) ? inside : path
      }
      const allowed = async (path, what) => {
        if (!(await mayWrite(policy, path))) throw new Error(deniedText(policy, what, pathShown(path)))
      }
      const file = args.file ? resolve(cwd, args.file) : undefined
      if (file) await mustBeFile(cwd, args.file, 'file')
      // an image that is not there is left out, and said; a call none of whose images is there is refused
      let images
      if (Array.isArray(args.images) && args.images.length) {
        const found = []
        const missing = []
        for (const path of args.images) ((await kindAt(resolve(cwd, String(path)))) === 'file' ? found : missing).push(String(path))
        if (found.length === 0) throw new Error(await noFile(cwd, missing[0], `image${missing.length > 1 ? ` (and ${missing.length - 1} more)` : ''}`))
        if (missing.length) notes.push(`left out ${missing.length} image(s) that do not exist: ${listed(missing, 6)}`)
        images = found.map(path => resolve(cwd, path))
      }
      const sourcePaths = (Array.isArray(args.sources) ? args.sources : typeof args.sources === 'string' ? [args.sources] : []).filter(Boolean)
      for (const path of sourcePaths) await mustBeFile(cwd, path, 'source')
      // Every file the call reads, which nothing it writes may land on: the same path, another letter case of it, a
      // link or a hard link to it.
      const inputs = [file, ...(images ?? []), ...sourcePaths.map(path => resolve(cwd, path))].filter(Boolean)
      const inputNamed = input => (input === file ? 'the input file' : images?.includes(input) ? `the image ${pathShown(input)}` : `the source ${pathShown(input)}`)
      const landsOn = async (target, why) => {
        const input = await inputAt(target, inputs)
        if (!input) return
        const alias = resolve(target) === input ? '' : ` (${pathShown(target)} is that file under another name)`
        throw new Error(`${why}: give it a path other than ${inputNamed(input)}${alias}`)
      }
      // the results and the redacted copy each take a file of their own, never an input's
      let output = resolve(cwd, args.output || defaultOutput(cwd, file, images))
      if ((await kindAt(output)) === 'directory' || /[\\/]$/.test(args.output ?? '')) output = join(output, basename(defaultOutput(cwd, file, images)))
      await landsOn(output, 'output is where the results go')
      let copies = !readOnly && args.redact !== undefined && args.redact !== false && args.redact !== 'false' && args.redact !== null && args.redact !== ''
      const copyTo = !copies ? undefined : typeof args.redact === 'string' && args.redact !== 'true'
        ? resolve(cwd, args.redact)
        : file ? `${file.slice(0, file.length - extname(file).length)}.redacted${extname(file)}` : resolve(cwd, 'system-one-redacted.txt')
      if (copyTo) await landsOn(copyTo, 'redact writes a copy')
      // one path for both: the copy, which was asked for by name, takes it, and the results go where they go by default
      if (copyTo && (await sameFile(copyTo, output))) {
        const fallback = resolve(cwd, defaultOutput(cwd, file, images))
        if ((await sameFile(fallback, copyTo)) || (await inputAt(fallback, inputs))) throw new Error('redact writes a copy beside the results: give it a path other than output')
        output = fallback
        notes.push(`output and redact named one file: the redacted copy is written there, the results to ${relative(cwd, output) || output}`)
      }
      if (readOnly) output = undefined
      const items = await loadItems({
        file,
        items: args.items,
        images,
        textField: args.text_field,
        idField: args.id_field,
        maxItems: config.toolMaxItems,
        maxChars: config.toolMaxChars,
        cwd,
        notes,
      })
      // what System One reads of an item longer than it reads is said, not cut unseen
      const long = items.filter(item => item.cut).length
      if (long) notes.push(`${long} item(s) are longer than ${config.toolMaxChars} characters: System One read the first ${config.toolMaxChars} of each`)
      // the images come first, then the texts (loadItems)
      const imageCount = items.filter(item => item.images).length
      const hasImages = imageCount > 0
      const hasTexts = imageCount < items.length
      // a question these items cannot answer at all (a span question of images, a box question of text) is left out
      // when others remain, and said
      const cannot = Object.entries(questions).map(([id, question]) => [id, unanswerable(question, { images: hasImages, texts: hasTexts })]).filter(([, why]) => why)
      if (cannot.length && cannot.length < Object.keys(questions).length) {
        questions = Object.fromEntries(Object.entries(questions).filter(([id]) => !cannot.some(([gone]) => gone === id)))
        for (const [id, why] of cannot) notes.push(`not asked: ${id} (${why})`)
      }
      const grounded = sourcePaths.length > 0
      // a signal that judges a reply reads the records' own "user" and "answer" fields when state_fields names none
      const needs = [...new Set(Object.values(questions).flatMap(question => ROUTER_SIGNALS[question.signal]?.parts ?? []))]
      const own = needs.length > 0 && !grounded && items.every(item => typeof item.record === 'object' && item.record !== null && needs.every(part => item.record[part] !== undefined))
      const fields = fieldsOf(args.state_fields) ?? (own ? Object.fromEntries(needs.map(part => [part, part])) : undefined)
      if ((fields || grounded) && hasImages) throw new Error('sources and state_fields read text items, not images')
      const parts = fields || grounded ? [...new Set([...Object.keys(fields ?? {}), ...(grounded ? ['context', 'answer'] : [])])] : undefined
      // a redacted copy replaces what pii or the call's own labels mark in each item's text: without either, it is
      // personal data; where no copy can be made, the questions are answered and the copy is left out, and said
      if (copies) {
        const marking = Object.values(questions).filter(question => kindOf(question) === 'span' && !question.signal && question.preset !== 'halu' && question.preset !== 'evidence')
        const cut = items.find(item => item.cut)
        const none = hasImages ? (hasTexts ? 'the items include images' : 'the items are images')
          : parts ? 'halu, evidence, sources and state_fields mark words against sources, not words to replace'
            : cut ? `item ${cut.id} is longer than ${config.toolMaxChars} characters, and a copy needs each item whole` : undefined
        if (none) {
          copies = false
          notes.push(`no redacted copy: ${none}`)
        } else if (marking.length === 0) {
          // an id whose columns no question's answers take
          const taken = new Set([...ROW_COLUMNS, ...Object.keys(questions).flatMap(name => [name, ...['labels', 'spans', 'confidence', 'yes'].map(suffix => `${name}_${suffix}`)])])
          let id = 'pii'
          while ([id, `${id}_labels`, `${id}_spans`].some(column => taken.has(column))) id = `${id}_`
          questions = { ...questions, [id]: { preset: 'pii' } }
          notes.push(`redact: no pii or span question said what to replace, so ${id} ({"preset": "pii"}) marks the personal data it replaces`)
        }
      }
      const model = args.model || config.model
      const spanModel = args.span_model || config.spanModel
      // Each kind of item is asked what it can answer: an image the choice, noul, score and box questions, a text all
      // but box questions. A call of one kind is planned whole, so that a question it cannot answer is refused, with how.
      const kinds = [hasImages && { images: true, from: 0, to: imageCount }, hasTexts && { images: false, from: imageCount, to: items.length }].filter(Boolean)
      const plans = kinds.map(kind => {
        const fit = kinds.length > 1 ? Object.fromEntries(Object.entries(questions).filter(([, question]) => !unanswerable(question, { images: kind.images }))) : questions
        return { ...kind, ...planQuestions(fit, { model, spanModel, parts, images: kind.images }) }
      })
      const asked = {}
      for (const plan of plans) for (const [id, entry] of Object.entries(plan.asked)) if (!Object.hasOwn(asked, id)) put(asked, id, entry)
      const groups = plans.flatMap(plan => plan.groups.map(group => ({ ...group, images: plan.images, from: plan.from, to: plan.to })))
      // whether a question was asked of the item at `index`, as a box question is not of a text in a call with images
      const applies = (id, index) => plans.some(plan => index >= plan.from && index < plan.to && Object.hasOwn(plan.asked, id))
      const unaskedOf = index => new Set(Object.keys(questions).filter(id => !applies(id, index)))
      // each answer takes a column of its own in a CSV and in the first rows, beside the item's id
      const clash = columnClash(questions, asked)
      if (clash) throw new Error(clash)
      // what gets masked (personal data, else what a redacted copy replaces) keeps the spans that overlap, so that their
      // union is masked: a name marked inside an email address hides the whole address
      const marks = Object.keys(questions).filter(id => asked[id].kind === 'span' && questions[id].preset !== 'halu' && questions[id].preset !== 'evidence')
      const personalMarks = marks.filter(id => questions[id].preset === 'pii')
      const masked = new Set(copies && personalMarks.length === 0 ? marks : personalMarks)
      const boxIds = Object.entries(questions).filter(([, question]) => question.type === 'box').map(([id]) => id)
      // where the annotated copies go: beside each image, into a folder, or (for one image) into a file
      let annotateTo
      if (!readOnly && args.annotate !== undefined && args.annotate !== null && args.annotate !== false && args.annotate !== 'false' && args.annotate !== '') {
        if (!hasImages || boxIds.length === 0) {
          notes.push('nothing annotated: annotate draws the boxes that box questions find on images; add one, as {"where": {"type": "box", "instructions": "the spike, drop or flatline"}}')
        } else if (args.annotate === true || args.annotate === 'true') {
          annotateTo = { beside: true }
        } else {
          const to = resolve(cwd, String(args.annotate))
          const at = await kindAt(to)
          if (/\.png$/i.test(to) && items.length === 1 && at !== 'directory') annotateTo = { file: to }
          else if (at === 'file') throw new Error(`annotate names a folder for the annotated copies, and ${args.annotate} is a file: name a folder, as "metrics/annotated/", or pass true for a copy beside each image`)
          else annotateTo = { dir: to }
        }
      }
      if (output) await allowed(output, 'output')
      if (copies) await allowed(copyTo, 'redact')
      // each image's annotated copy, decided before anything is asked
      const drawTo = new Map()
      if (annotateTo) {
        const kept = [...inputs, ...(output ? [output] : []), ...(copies ? [copyTo] : [])]
        for (const item of items) {
          if (!item.path) continue
          const target = await annotatedPath(annotateTo, item, kept)
          await allowed(target, 'annotate')
          drawTo.set(item, target)
        }
      }
      const context = args.context?.trim()
      const sources = grounded ? await readSources(sourcePaths, cwd, context) : undefined

      // a field read for state_fields is cut at toolMaxChars, as an item is, and said
      let fieldsCut = 0
      const states = items.map(item => {
        if (!parts) {
          // an image is what its state is about; an empty text says so, and is no image
          const read = [context, item.text].filter(Boolean).join('\n\n')
          return { state: read || (item.images ? 'An image.' : '(empty)'), shift: context && item.text ? context.length + 2 : 0, images: item.images }
        }
        const state = {}
        if (fields) {
          if (typeof item.record !== 'object' || item.record === null) throw new Error('state_fields reads the fields of records: pass a .jsonl, .json or .csv file')
          for (const [part, field] of Object.entries(fields)) {
            const value = item.record[field]
            if (value === undefined) throw new Error(`state_fields: "${field}" is not a field of item ${item.id}`)
            const read = typeof value === 'string' ? value : JSON.stringify(value)
            if (read.length > config.toolMaxChars) fieldsCut++
            state[part] = read.slice(0, config.toolMaxChars)
          }
        }
        if (grounded) {
          // a record's own context, as the customer's message a reply answers, reads before the sources
          const own = state.context === undefined ? '' : `${state.context}\n\n`
          state.context = own + sources.context
          state.answer ??= item.text
          return { state, shift: 0, sourcesAt: own.length }
        }
        return { state, shift: 0 }
      })
      if (fieldsCut) notes.push(`${fieldsCut} field value(s) are longer than ${config.toolMaxChars} characters: System One read the first ${config.toolMaxChars} of each`)
      // each request stays within the Decision API's estimate, which counts an item's state once for every question, and
      // within the questions one request may ask
      const jobs = []
      for (const group of groups) {
        const asks = Object.keys(group.questions).length
        if (asks > QUESTIONS_PER_REQUEST) throw new Error(`${asks} questions of ${group.model} are more than one request may ask (${QUESTIONS_PER_REQUEST}): ask fewer questions per call`)
        // a group of signals that read one part of each item reads that part as its text
        const texts = group.reads ? states.map(entry => ({ ...entry, state: String(entry.state[group.reads] ?? '') })) : states
        const kindOfItems = texts.slice(group.from, group.to)
        if (!group.images) {
          const heaviest = Math.max(...kindOfItems.map(({ state }) => stateCost(state)))
          if (heaviest * asks > REQUEST_TOKENS) {
            throw new Error(`${asks} question(s) of ${group.model} over an item of about ${heaviest} bytes are more than one request may read (${REQUEST_TOKENS} tokens, each question reading the item again): ask fewer questions per call, or pass shorter items or sources`)
          }
        }
        // an image goes in a request of its own
        const slices = group.images ? kindOfItems.map((_, start) => ({ start, count: 1 })) : requestSlices(kindOfItems.map(({ state }) => state), asks)
        for (const { start, count } of slices) jobs.push({ group, start: group.from + start, count, texts })
      }
      const started = performance.now()
      // Each request is settled on its own: the items of one that failed carry its error in place of their answers, and
      // the others keep theirs. System One unavailable after a request's retries stops the requests still waiting or
      // running, which would only wait out the same retries on the same client.
      const stop = new AbortController()
      const signal = exec.signal ? AbortSignal.any([exec.signal, stop.signal]) : stop.signal
      let stopped
      const settled = await Promise.all(jobs.map(async job => {
        const slice = job.texts.slice(job.start, job.start + job.count).map(({ state, images: attached }) => ({ state, images: attached }))
        try {
          return { ...job, response: await client.ask({ model: job.group.model, questions: job.group.questions, inputs: slice, signal, batch: true, ...BATCH_LIMITS }) }
        } catch (error) {
          if (stopped && stop.signal.aborted && !exec.signal?.aborted) return { ...job, error: stopped, skipped: true }
          if (!stopped && unavailable(error)) {
            stopped = error
            stop.abort(error)
          }
          return { ...job, error }
        }
      }))
      if (exec.signal?.aborted) throw settled.find(job => job.error)?.error ?? exec.signal.reason
      const answered = settled.filter(job => job.response)
      const failed = settled.filter(job => job.error)
      if (answered.length === 0) {
        const error = (failed.find(job => !job.skipped) ?? failed[0]).error
        throw unavailable(error) ? new Error(unavailableText(error)) : error
      }
      const seconds = Math.round((performance.now() - started) / 10) / 100

      const rows = items.map(item => ({ id: item.id, text: item.source ?? clip(item.text, 200), answers: {} }))
      // the items of a request that failed: each of its questions answered by the error
      for (const job of failed) {
        const said = clip(String(job.error?.message ?? job.error), 200)
        const answer = job.skipped ? { error: 'not_asked', reason: `another request of this call failed first: ${said}` } : { error: unavailable(job.error) ? 'unavailable' : 'request_failed', reason: said }
        for (let index = job.start; index < job.start + job.count; index++) {
          for (const id of Object.values(job.group.ids)) put(rows[index].answers, id, { ...answer })
        }
      }
      let partial
      if (failed.length) {
        const first = failed.find(job => !job.skipped) ?? failed[0]
        const hit = new Set(failed.flatMap(job => Array.from({ length: job.count }, (_, offset) => job.start + offset)))
        partial = { failed: failed.length, requests: jobs.length, items: hit.size }
        notes.push(`${failed.length} of ${jobs.length} requests failed (${clip(String(first.error?.message ?? first.error), 160)}): ${hit.size} item(s) have an error in place of their answers; call again for them`)
      }
      // the spans of each item, whole, for redaction and Chat: rows keep no personal data's text
      const found = items.map(() => ({}))
      // the items whose own id is personal data, by the label it carries
      const idLabels = new Map()
      let inputTokens = 0
      for (const { group, start, response } of answered) {
        inputTokens += response.inputTokens
        response.answers.forEach((answers, offset) => {
          const index = start + offset
          // an answer and its spans under the question's own id only: an id such as __proto__ names no other field
          const given = (record, wire) => (record && Object.hasOwn(record, wire) ? record[wire] : undefined)
          for (const [wire, id] of Object.entries(group.ids)) {
            const entry = asked[id]
            const answer = given(answers, wire)
            const marked = given(response.spans[offset], wire)
            if (entry.kind !== 'span') {
              const compact = compactAnswer(answer)
              const base = questions[id].type === 'score' ? levelBase(questions[id].criteria) : 0
              // a box is found or not by the confidence as the model gave it, before it is rounded
              const box = questions[id].type === 'box' && compact.box ? { found: answer.confidence >= FOUND_AT } : {}
              put(rows[index].answers, id, base && typeof compact.score === 'number' ? { ...compact, score: round3(compact.score + base) } : { ...compact, ...box })
              continue
            }
            if (answer?.error || (!answer && !Array.isArray(marked))) {
              put(rows[index].answers, id, compactAnswer(answer))
              continue
            }
            const { state, shift } = states[index]
            const read = typeof state === 'string' ? state : state[entry.over] ?? ''
            const question = questions[id]
            let spans = placeSpans(marked, read, { min: question.threshold ?? (question.preset === 'halu' ? HALU_AT : question.preset === 'pii' ? config.privacyAt ?? PII_AT : 0), overlaps: masked.has(id) })
              .filter(span => span.start >= shift)
              .map(span => ({ ...span, start: span.start - shift, end: span.end - shift }))
            const kept = question.preset !== 'pii' || question.labels === 'all' ? undefined : question.labels ?? config.privacyLabels ?? SENSITIVE_LABELS
            if (kept) spans = spans.filter(span => kept.includes(span.label))
            if (question.preset === 'pii') spans = spans.map(span => trimSpan(span, items[index].text)).filter(Boolean)
            // An item's id names it, whatever it looks like. What a hidden question (personal data, or what a redacted
            // copy replaces) marks in it is said, and the id is named by a pseudonym.
            const named = typeof state === 'string' ? items[index].segments?.find(segment => segment.field !== undefined && segment.field === items[index].idField) : undefined
            const inId = span => named && span.start < named.end && span.end > named.start
            if (masked.has(id) && !idLabels.has(index)) {
              const own = spans.find(span => inId(span) && (question.preset !== 'pii' || !notPersonal(span, items[index].text)))
              if (own) idLabels.set(index, own.label)
            }
            spans = spans.filter(span => !inId(span) && !(question.preset === 'pii' && notPersonal(span, items[index].text)))
            const at = states[index].sourcesAt
            if (question.preset === 'evidence' && at !== undefined && entry.over === 'context') {
              spans = spans.map(span => ({ ...span, ...(span.start >= at ? sourceAt(sources.blocks, span.start - at) : { source: fields.context }) }))
            }
            put(found[index], id, spans)
            put(rows[index].answers, id, {
              n: spans.length,
              p: round3(Math.max(0, ...spans.map(span => span.probability))),
              labels: labelCounts(spans),
              spans: spans.map(span => ({
                label: span.label,
                start: span.start,
                end: span.end,
                ...(masked.has(id) ? {} : { text: clip(span.text, 300) }),
                p: round3(span.probability),
                ...(span.source ? { source: span.source, line: span.line } : {}),
              })),
            })
          }
        })
      }
      // What a hidden question marks (personal data, or what a redacted copy replaces) stays out of the results' text as
      // well, and an item one of them could not look at keeps no text there.
      const personalIds = Object.keys(questions).filter(id => questions[id].preset === 'pii')
      const hiddenIds = [...masked]
      const hiddenOf = index => hiddenIds.flatMap(id => found[index][id] ?? [])
      const unlooked = index => hiddenIds.some(id => applies(id, index) && rows[index].answers[id]?.error !== undefined)
      for (const [index, item] of items.entries()) {
        const spans = hiddenOf(index)
        if (unlooked(index)) delete rows[index].text
        else if (spans.length && item.source === undefined) rows[index].text = clip(redactItem(item, spans).text, 200)
        // another question's marks hold what a hidden question marked inside them by its label only, in the results
        // and in the summary's examples
        if (!spans.length) continue
        for (const id of marks.filter(own => !masked.has(own))) {
          const shown = found[index][id]
          if (!shown?.length || asked[id].over) continue
          const moved = maskAround(item.text, shown, joined(everywhere(item.text, spans))).spans
          rows[index].answers[id].spans.forEach((span, at) => {
            span.text = clip(moved[at].text, 300)
          })
        }
      }
      // and one that looks for jailbreaks or harm keeps their words out: System Two reading the results reads no attack
      if (Object.values(questions).some(question => SAFETY_SIGNALS.has(question.signal))) for (const row of rows) delete row.text

      // Item ids that are personal data, or that hold what a redacted copy replaces: what a hidden question marked in an
      // id's own field, or, in a call that looks for personal data, an id shaped as an email address, an IP address or
      // a phone number. This result and the redacted copy name them by stable pseudonyms; the results file keeps each id
      // beside its pseudonym, for you to act on.
      let pseudonyms
      // a JSON object's keys that are personal data, named the same way in its redacted copy
      const keyAliases = []
      const on = config.pseudonymIds !== false
      if (personalIds.length) {
        for (const [index, item] of items.entries()) {
          if (!idLabels.has(index) && item.idField !== undefined && personalShape(item.id)) idLabels.set(index, personalShape(item.id))
        }
      }
      const keyLabels = personalIds.length ? items.map(item => (item.key !== undefined ? personalShape(item.key) : undefined)) : []
      if (on && (idLabels.size || keyLabels.some(Boolean))) {
        const key = await pseudonymKey()
        for (const [index, label] of idLabels) rows[index].pseudonym = pseudonym(key, label, items[index].id)
        keyLabels.forEach((label, index) => {
          if (label) keyAliases[index] = pseudonym(key, label, items[index].key)
        })
      }
      if (idLabels.size) {
        const labels = labelCounts([...idLabels.values()].map(label => ({ label })))
        const said = Object.entries(labels).map(([label, n]) => `${label} ${n}`).join(', ')
        pseudonyms = { items: idLabels.size, labels, named: on ? 'pseudonyms' : 'kept' }
        notes.push(on
          ? `${idLabels.size} item id(s) are personal data (${said}): this result and the redacted copy name them by pseudonyms, as ${rows.find(row => row.pseudonym).pseudonym}, the same for the same id in every call; the results file keeps each id beside its pseudonym`
          : `${idLabels.size} item id(s) are personal data (${said}), and they are kept as they are in this result, the results file and the redacted copy (pseudonymIds is off)`)
      }
      // what this result shows of each item: its pseudonym where it has one
      const shownRows = rows.map(({ pseudonym: alias, ...row }) => (alias ? { ...row, id: alias } : row))
      // what a CSV and the first rows show of each row: no words of a hidden question, no cell for a question not asked
      const show = index => ({ hidden: masked, unasked: unaskedOf(index) })

      // The results first: a copy or an image that fails later leaves them written.
      if (output) {
        await allowed(output, 'output')
        await mkdir(dirname(output), { recursive: true })
        await writeResults(output, rows, questions, asked, show)
      }

      let redacted
      if (copies) {
        // personal data when the call looks for it, else what its own labels mark; a copy of items some of which were
        // not looked at would pass on what they hold, so none is written
        const marking = [...masked]
        const unmarked = items.map((_, index) => index).filter(index => marking.some(id => rows[index].answers[id]?.error !== undefined))
        if (unmarked.length) {
          const why = rows[unmarked[0]].answers[marking.find(id => rows[unmarked[0]].answers[id]?.error !== undefined)]
          notes.push(`no redacted copy: ${unmarked.length} item(s) were not marked (${clip([why.error, why.reason].filter(Boolean).join(': '), 160)}), and a copy would keep what they hold; call again`)
          copies = false
        }
      }
      if (copies) {
        const marking = [...masked]
        const out = copyTo
        // A name in a record that holds what the copy replaces (a field's, as {"alice@example.test": "hello"} has, or a
        // key's inside a value) takes its pseudonym, as an id does, so that two such names stay two; with pseudonymIds
        // off it stays as it is, and the result says so.
        const copiesOf = rename => items.map((item, index) => {
          const spans = marking.flatMap(id => found[index][id] ?? [])
          return spans.length ? redactItem(item, spans, { rename }) : { record: item.record, replaced: 0 }
        })
        let names = 0
        let copied = copiesOf(name => {
          names++
          return name
        })
        if (names && on) {
          const key = await pseudonymKey()
          copied = copiesOf((name, label) => pseudonym(key, label, name))
        } else if (names) notes.push(`${names} field name(s) hold what the copy replaces, and they are kept as they are in the redacted copy (pseudonymIds is off)`)
        let spanCount = 0
        let changed = 0
        const records = items.map((item, index) => {
          const copy = copied[index]
          spanCount += copy.replaced
          if (copy.replaced) changed++
          const alias = rows[index].pseudonym
          const record = copy.record
          return alias && typeof record === 'object' && record !== null && item.idField !== undefined ? { ...record, [item.idField]: alias } : record
        })
        await allowed(out, 'redact')
        await mkdir(dirname(out), { recursive: true })
        await writeRedacted({ file, format: file ? formatOf(file) : 'lines', output: out, items, records, keys: keyAliases.length ? items.map((item, index) => keyAliases[index] ?? item.key) : undefined })
        redacted = { output: out, items: changed, spans: spanCount }
      }

      // An image the annotator cannot draw on (not a PNG, a 16-bit or an interlaced one) is left out, and said; its
      // boxes are in the results.
      let annotated
      if (annotateTo) {
        const dir = annotateTo.dir ?? dirname(annotateTo.file ?? items[0].path)
        await allowed(dir, 'annotate')
        await mkdir(dir, { recursive: true })
        annotated = { dir, files: [], skipped: [] }
        for (const [index, item] of items.entries()) {
          const boxes = boxIds.map(id => rows[index].answers[id]).filter(a => a?.box && a.found)
          if (!item.path || boxes.length === 0) continue
          const bytes = await readFile(item.path)
          let drawn
          const why = undrawable(bytes)
          if (!why) {
            try {
              drawn = annotatePng(bytes, boxes.map(a => ({ box: a.box })))
            } catch (error) {
              annotated.skipped.push(`${item.source} (${clip(String(error?.message ?? error), 100)})`)
              continue
            }
          }
          if (!drawn) {
            annotated.skipped.push(`${item.source} (${why})`)
            continue
          }
          const out = drawTo.get(item)
          await allowed(out, 'annotate')
          await mkdir(dirname(out), { recursive: true })
          await writeFile(out, drawn)
          annotated.files.push(relative(cwd, out) || out)
        }
      }
      const summary = summarize(questions, shownRows, asked, { hidden: masked, applies })

      // for Chat: a few marked items per span question, and the images with their boxes
      const examples = {}
      for (const [id, question] of Object.entries(questions)) {
        if (asked[id].kind !== 'span') continue
        // what a hidden question marks shows by its label only, as personal data does
        const kind = masked.has(id) ? 'masked' : question.preset === 'halu' ? 'unsupported' : question.preset === 'evidence' ? 'evidence' : 'labels'
        const picked = items.map((item, index) => ({ item, index, spans: found[index][id] ?? [] }))
          .filter(entry => entry.spans.length)
          .sort((x, y) => Math.max(...y.spans.map(span => span.probability)) - Math.max(...x.spans.map(span => span.probability)))
          .slice(0, 3)
        if (picked.length === 0) continue
        put(examples, id, picked.map(({ item, index, spans }) => {
          let read = asked[id].over && typeof states[index].state === 'object' ? states[index].state[asked[id].over] : item.text
          // an id named by a pseudonym is not shown in its own item's text either
          const alias = rows[index].pseudonym && item.segments?.find(segment => segment.field !== undefined && segment.field === item.idField)
          const ids = alias && read === item.text ? [{ label: idLabels.get(index), start: alias.start, end: alias.end, text: read.slice(alias.start, alias.end), probability: 1 }] : []
          // personal data wherever it recurs, as the redacted copy and the results file have it
          let shown = kind === 'masked' ? joined(everywhere(read, [...spans, ...ids])) : apart(spans)
          // and what a hidden question marked, masked in another question's excerpt, inside its marks too
          const hidden = kind === 'labels' ? [...hiddenOf(index), ...ids] : []
          if (hidden.length) ({ text: read, spans: shown } = maskAround(read, shown, read === item.text ? joined(everywhere(read, hidden)) : hidden))
          // evidence reads as a passage of its own file, or of the record's own context
          const block = kind === 'evidence' && spans[0].line !== undefined ? sources?.blocks.find(entry => entry.source === spans[0].source) : undefined
          if (block) {
            const start = states[index].sourcesAt + block.start
            read = block.text
            shown = spans.filter(span => span.source === block.source && span.line !== undefined).map(span => ({ ...span, start: span.start - start, end: span.end - start }))
          } else if (kind === 'evidence' && spans[0].source !== undefined) {
            read = read.slice(0, states[index].sourcesAt).trimEnd()
            shown = spans.filter(span => span.line === undefined && span.source === spans[0].source)
          }
          return {
            id: rows[index].pseudonym ?? item.id,
            kind,
            parts: excerpts(read, shown, { hide: kind === 'masked', around: kind === 'evidence' ? 28 : 40, max: 1, clip: 120, rank: true })[0],
            ...(spans[0].source ? { source: placeOf(spans[0]) } : {}),
          }
        }))
      }
      const shownImages = hasImages
        ? items.map((item, index) => ({ item, index })).filter(({ item }) => item.images).slice(0, IMAGES_SHOWN).map(({ item, index }) => ({
          id: item.id,
          path: item.path,
          boxes: boxIds.map(id => ({ id, ...rows[index].answers[id] })).filter(a => a.box && a.found).map(({ id, box, confidence }) => ({ id, box, confidence })),
          answers: Object.fromEntries(Object.entries(rows[index].answers).filter(([id]) => questions[id].type !== 'box').slice(0, 3)),
        }))
        : undefined
      const used = [...new Set(groups.map(group => group.model))]
      const value = {
        model: used.includes(model) ? model : used[0],
        ...(used.length > 1 ? { spanModel } : {}),
        items: items.length,
        questions: Object.keys(questions).length,
        requests: jobs.length,
        seconds,
        inputTokens,
        output,
        summary,
        preview: shownRows.slice(0, 3).map((row, index) => flatRow(row, questions, asked, show(index))),
        ...(Object.keys(examples).length ? { examples } : {}),
        ...(shownImages ? { images: shownImages } : {}),
        ...(redacted ? { redacted } : {}),
        ...(annotated ? { annotated } : {}),
        ...(sources ? { sources: sources.blocks.map(block => block.source) } : {}),
        ...(pseudonyms ? { pseudonyms } : {}),
        ...(partial ? { partial } : {}),
        ...(notes.length ? { notes } : {}),
      }
      trace({
        ev: 's1',
        reflex: 'tool',
        session: exec.agent?.session?.id,
        model: value.model,
        spanModel: value.spanModel,
        items: items.length,
        requests: jobs.length,
        ms: Math.round(seconds * 1000),
        inputTokens,
        questions,
        summary,
        output,
      })
      return value
    },
  })
}
