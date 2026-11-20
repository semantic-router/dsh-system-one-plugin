/**
 * Append-only JSONL trace of what System One decided and what the agent did,
 * one record per line, readable by its owner only. Every string is redacted
 * before it is encoded, so no key, token, password, cookie or authorization
 * header reaches the file; a trace an earlier version wrote is scrubbed the
 * same way when the trace opens. Records carry questions, probabilities,
 * verdicts, and truncated tool arguments.
 */
import { randomBytes } from 'node:crypto'
import { appendFileSync, createReadStream, createWriteStream } from 'node:fs'
import { chmod, mkdir, open, rename, rm } from 'node:fs/promises'
import { once } from 'node:events'
import { dirname } from 'node:path'
import { createInterface } from 'node:readline'

const REDACTED = '[redacted]'
// a secret starts where no letter or digit comes before it: a line break, a quote, or a JSON escape such as \n
const START = String.raw`(?<=^|[^A-Za-z0-9]|\\[nrtbf])`
// keys and tokens known by their prefix; a value cut short by clip (… (+N chars)) is redacted whatever is left of it
const PREFIX = String.raw`(?:sk-|sk_live_|sk_test_|rk_live_|rk_test_|hf_|gh[pousr]_|github_pat_|npm_|glpat-|xox[abposr]-|AKIA|ASIA|AIza|eyJ)`
// a field, a variable or a flag whose name ends in one of these words holds a secret: DECISION_API_KEY, github_token,
// "password", x-api-key; max_tokens and apiKeyEnv do not
const SECRET_WORD = String.raw`(?:api[-_]?key|apikey|access[-_]?key|secret[-_]?key|private[-_]?key|secret|token|passw(?:or)?d|passphrase|credentials?)`

const SECRET_SHAPES = [
  [/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/g, REDACTED],
  // a header's whole value: Authorization: Bearer …, Cookie: a=1; b=2
  [/(?<![A-Za-z0-9-])((?:proxy-)?authorization|x-api-key|api-key|x-auth-token|x-access-token|x-goog-api-key|cookie|set-cookie)(["']?[ \t]*:[ \t]*["']?)[^\r\n"']+/gi, `$1$2${REDACTED}`],
  [new RegExp(String.raw`(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]*?${SECRET_WORD})(["']?[ \t]*[:=][ \t]*["']?)[^\s"',;&}\]\\]+`, 'gi'), `$1$2${REDACTED}`],
  [new RegExp(String.raw`(?<![A-Za-z0-9])(--?(?:api[-_]?key|token|access[-_]?token|auth[-_]?token|password|passwd|secret|client[-_]?secret))([ \t]+|=)(["']?)[^\s"']+`, 'gi'), `$1$2$3${REDACTED}`],
  [new RegExp(String.raw`${START}(?:sk-[A-Za-z0-9][A-Za-z0-9_-]{15,}|(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|npm_[A-Za-z0-9]{30,}|glpat-[A-Za-z0-9_-]{20,}|xox[abposr]-[A-Za-z0-9-]{10,}|(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Z])|AIza[0-9A-Za-z_-]{35}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})`, 'g'), REDACTED],
  [/(?<![A-Za-z0-9])(bearer)\s+(?=[A-Za-z0-9._~+/=-]*[0-9._~+/=-])[A-Za-z0-9._~+/=-]{8,}/gi, `$1 ${REDACTED}`],
  // credentials in a URL, and the query parameters that carry one
  [/(?<=[a-z][a-z0-9+.-]*:\/\/)[^/\s:@"']+:[^/\s@"']+(?=@)/gi, REDACTED],
  [/([?&](?:api[-_]?key|apikey|key|token|access[-_]?token|auth|password|secret|client_secret|sig|signature)=)[^&\s"'#]+/gi, `$1${REDACTED}`],
  [new RegExp(String.raw`${START}${PREFIX}[A-Za-z0-9_.-]*(?=…)`, 'g'), REDACTED],
]

export function redact(text) {
  let out = String(text)
  for (const [shape, replacement] of SECRET_SHAPES) out = out.replace(shape, replacement)
  return out
}

// a field of a record whose whole value is a secret
const SECRET_FIELD = new RegExp(String.raw`^(?:(?:proxy-)?authorization|cookie|set-cookie|x-api-key)$|(?:^|[-_.])${SECRET_WORD}$`, 'i')

/** JSON.stringify's replacer for anything written to disk: each string redacted before it is encoded. */
export function scrubbed(key, value) {
  if (typeof value === 'string') return value && SECRET_FIELD.test(key) ? REDACTED : redact(value)
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).some(name => redact(name) !== name)) {
    return Object.fromEntries(Object.entries(value).map(([name, inner]) => [redact(name), inner]))
  }
  return value
}

export function clip(text, max) {
  const value = String(text ?? '')
  return value.length <= max ? value : `${value.slice(0, max)}… (+${value.length - max} chars)`
}

// The trace's first line says how it was redacted; a trace without it, or with an older one, is scrubbed when it opens.
export const REDACTION = 1
const header = () => JSON.stringify({ t: new Date().toISOString(), ev: 'trace', redaction: REDACTION })

function scrubLine(line) {
  try {
    return JSON.stringify(JSON.parse(line), scrubbed)
  } catch {
    return redact(line)
  }
}

async function firstLine(path) {
  const handle = await open(path, 'r')
  try {
    const { buffer, bytesRead } = await handle.read({ buffer: Buffer.alloc(4096), position: 0 })
    return buffer.subarray(0, bytesRead).toString('utf8').split('\n')[0]
  } finally {
    await handle.close()
  }
}

function redactedAlready(line) {
  try {
    const head = JSON.parse(line)
    return head?.ev === 'trace' && head.redaction >= REDACTION
  } catch {
    return false
  }
}

// the scrubbed lines of `scratch` written over `path` itself, for a file that cannot be replaced
async function overwrite(path, scratch) {
  const handle = await open(path, 'r+')
  try {
    let position = 0
    for await (const chunk of createReadStream(scratch)) {
      await handle.write(chunk, 0, chunk.length, position)
      position += chunk.length
    }
    await handle.truncate(position)
  } finally {
    await handle.close()
  }
}

/**
 * The trace file, owner-only: a new one starts with its header, an older one is scrubbed: into a scratch file that
 * takes its place, or, where it cannot (a program holds it open without letting it be replaced, as Windows allows),
 * written over it.
 */
async function prepare(path) {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  let first
  try {
    first = await firstLine(path)
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error
    appendFileSync(path, `${header()}\n`, { mode: 0o600 })
    return
  }
  await chmod(path, 0o600)
  if (redactedAlready(first)) return
  const scratch = `${path}.${randomBytes(6).toString('hex')}.scrub`
  try {
    const out = createWriteStream(scratch, { flags: 'wx', mode: 0o600 })
    const closed = once(out, 'close')
    const write = async text => {
      if (!out.write(text)) await once(out, 'drain')
    }
    await write(`${header()}\n`)
    for await (const line of createInterface({ input: createReadStream(path, 'utf8'), crlfDelay: Infinity })) {
      if (line) await write(`${scrubLine(line)}\n`)
    }
    out.end()
    await closed
    await chmod(scratch, 0o600)
    try {
      await rename(scratch, path)
    } catch (error) {
      await overwrite(path, scratch).catch(() => {
        throw error
      })
    }
  } finally {
    await rm(scratch, { force: true })
  }
}

/**
 * @param {string | undefined} path - no path disables the trace
 * @returns {((entry: object) => void) & {ready: Promise<void>}} records written once the file is prepared; a trace an
 *   earlier version wrote that could not be scrubbed gets none, and the trace stops with a warning
 */
export function createTrace(path) {
  if (!path) return Object.assign(() => {}, { ready: Promise.resolve() })
  const waiting = []
  let prepared = false
  let off = false
  const append = line => {
    try {
      appendFileSync(path, line + '\n', { mode: 0o600 })
    } catch {
      // A trace write must never fail an agent step.
    }
  }
  const ready = prepare(path).catch(async error => {
    // a trace that still holds what an earlier version wrote, unscrubbed, is never written to
    const first = await firstLine(path).catch(() => undefined)
    if (first === undefined || redactedAlready(first)) return
    off = true
    process.stderr.write(`[System One] The trace is off: ${path} holds records an earlier version wrote, which may hold keys, and it could not be scrubbed (${error?.code ?? error?.message ?? error}). Close what holds it open and restart, or delete it.\n`)
  }).finally(() => {
    prepared = true
    const lines = waiting.splice(0)
    if (!off) for (const line of lines) append(line)
  })
  function record(entry) {
    if (off) return
    let line
    try {
      line = JSON.stringify({ t: new Date().toISOString(), ...entry }, scrubbed)
    } catch {
      return
    }
    if (prepared) append(line)
    else waiting.push(line)
  }
  record.ready = ready
  return record
}
