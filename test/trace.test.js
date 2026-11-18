import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { clip, createTrace, redact } from '../src/trace.js'

// the own part of each secret: none of it may reach a trace, wherever it sits and however it is written
const OWN = 'Qw7Er4Ty9Ui2Op5As8'
// put together here, so that no scanner takes this file for one holding a key
const KEY_BLOCK = kind => [`-----BEGIN ${kind} PRIVATE KEY-----`, `-----END ${kind} PRIVATE KEY-----`]
const SECRETS = [
  `the key\nsk-vsr-${OWN}`, `the key\\nsk-pt-${OWN}`, `ANTHROPIC_API_KEY=sk-ant-api03-${OWN}-Zx_9`, `export OPENAI_API_KEY=sk-proj-${OWN}_c-1`,
  `sk-${OWN}${OWN}`, `curl -H "Authorization: Bearer ${OWN}" -H 'x-api-key: ${OWN}'`, `authorization: Basic ${OWN}`,
  `Cookie: session=${OWN}; theme=dark`, `Set-Cookie: sid=${OWN}; Path=/; HttpOnly`, `{"api_key": "${OWN}", "n": 1}`, `{"client_secret":"${OWN}"}`,
  `token: ${OWN}`, `PGPASSWORD=${OWN} psql`, `aws_secret_access_key = ${OWN}`, `--api-key ${OWN}`, `--token=${OWN}`, `--password "${OWN}"`,
  `https://bob:${OWN}@db.example/x`, `GET /v1/models?key=${OWN}&q=1`, `hf_${OWN}xx`, `ghp_${OWN}xx`, `github_pat_${OWN}xx`, `npm_${OWN}${OWN}`,
  `glpat-${OWN}xx`, `xoxb-${OWN}`, `eyJhbGciOiJIUzI1NiJ9.eyJ${OWN}.${OWN}`, KEY_BLOCK('OPENSSH').join(`\n${OWN}\n`),
  `${KEY_BLOCK('RSA')[0]}\n${OWN}… (+900 chars)`, clip(`its key: sk-vsr-${OWN}`, 22), clip(`DECISION_API_KEY=sk-vsr-${OWN}`, 30),
]
// no eight characters in a row of it
function holdsNone(text) {
  for (let at = 0; at + 8 <= OWN.length; at++) if (text.includes(OWN.slice(at, at + 8))) return false
  return true
}

const scratch = async t => {
  const dir = await mkdtemp(join(tmpdir(), 's1-trace-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  return dir
}
const readLines = async path => (await readFile(path, 'utf8')).trim().split('\n').map(line => JSON.parse(line))

test('redaction leaves no part of a key, a token, a password, a cookie or an authorization header, wherever it sits', () => {
  for (const text of SECRETS) assert.ok(holdsNone(redact(text)), `${JSON.stringify(text)} became ${JSON.stringify(redact(text))}`)
  assert.equal(redact('curl -H "Authorization: Bearer abc.def-123" https://x'), 'curl -H "Authorization: [redacted]" https://x')
  assert.equal(redact('export OPENAI_API_KEY=sk-proj-abcdefghijklmnop1234 && run'), 'export OPENAI_API_KEY=[redacted] && run')
  assert.equal(redact(`see sk-vsr-${OWN}`.slice(0, 15) + '… (+20 chars)'), 'see [redacted]… (+20 chars)')
})

test('redaction keeps what only looks like a secret', () => {
  for (const text of ['--max_tokens=512', 'apiKeyEnv: DECISION_API_KEY', '"inputTokens": 1234', 'a basic implementation of the tokenizer', 'git push -u origin main',
    'task-management-dashboard-v2', 'risk-assessment-framework-version-two', 'scikit-learn and sk-learn', 'the key: press Enter', 'Bearer tokens expire']) {
    assert.equal(redact(text), text)
  }
})

test('a trace never holds a key: each string is redacted before it is encoded, in every field and every record', async t => {
  const path = join(await scratch(t), 'system-one', 'trace.jsonl')
  const trace = createTrace(path)
  trace({ ev: 'user', text: SECRETS.join('\n') })
  for (const [index, text] of SECRETS.entries()) trace({ ev: 'result', call: text, preview: clip(text, 40), nested: { list: [text], headers: { authorization: OWN, 'set-cookie': OWN, apiKey: OWN } }, [`key ${index}`]: text })
  await trace.ready
  const written = await readFile(path, 'utf8')
  assert.ok(holdsNone(written), 'no part of a secret in the file')
  const lines = await readLines(path)
  assert.equal(lines.length, 2 + SECRETS.length)
  assert.deepEqual([lines[0].ev, lines[0].redaction], ['trace', 1])
  assert.equal(lines[1].text.split('\n')[0], 'the key')
  assert.deepEqual(lines[2].nested.headers, { authorization: '[redacted]', 'set-cookie': '[redacted]', apiKey: '[redacted]' })
})

test('the trace is readable by its owner only, a file that was there before too', { skip: process.platform === 'win32' }, async t => {
  const dir = await scratch(t)
  const fresh = join(dir, 'new', 'trace.jsonl')
  const record = createTrace(fresh)
  record({ ev: 'x' })
  await record.ready
  assert.equal((await stat(fresh)).mode & 0o777, 0o600)
  assert.equal((await stat(join(dir, 'new'))).mode & 0o777, 0o700)

  const older = join(dir, 'older.jsonl')
  await writeFile(older, '{"t":"2026-11-12T00:00:00.000Z","ev":"user","text":"hello"}\n', { mode: 0o644 })
  const again = createTrace(older)
  again({ ev: 'x' })
  await again.ready
  assert.equal((await stat(older)).mode & 0o777, 0o600)
})

test('a trace an earlier version wrote is scrubbed once when it opens; records written meanwhile are kept', async t => {
  const path = join(await scratch(t), 'trace.jsonl')
  // as 0.8.0 wrote it: redacted after encoding, so a key after a line break or with dashes stayed
  const earlier = [
    JSON.stringify({ t: '2026-11-12T09:00:00.000Z', ev: 'result', preview: `.env:\nDECISION_API_KEY=sk-vsr-${OWN}` }),
    JSON.stringify({ t: '2026-11-12T09:00:01.000Z', ev: 'user', text: `use sk-ant-api03-${OWN}-x_y for it` }),
    '',
    `{"t":"2026-11-12T09:00:02.000Z","ev":"assistant","text":"done\\nsk-vsr-${OWN}`,
  ]
  await writeFile(path, `${earlier.join('\n')}\n`, { mode: 0o644 })
  const trace = createTrace(path)
  trace({ ev: 'user', text: 'while the scrub runs' })
  await trace.ready
  const written = await readFile(path, 'utf8')
  assert.ok(holdsNone(written), 'no part of a secret in the file')
  const lines = written.trim().split('\n')
  assert.equal(lines.length, 5)
  assert.deepEqual(lines.slice(0, 3).map(line => JSON.parse(line).ev), ['trace', 'result', 'user'])
  assert.equal(JSON.parse(lines[1]).preview, '.env:\nDECISION_API_KEY=[redacted]')
  assert.equal(JSON.parse(lines[2]).text, 'use [redacted] for it')
  assert.ok(lines[3].startsWith('{"t":"2026-11-12T09:00:02.000Z","ev":"assistant","text":"done\\n[redacted]'), 'a line cut short is redacted as text')
  assert.equal(JSON.parse(lines[4]).text, 'while the scrub runs')

  // a trace already scrubbed is left as it is
  const { ino } = await stat(path)
  const reopened = createTrace(path)
  reopened({ ev: 'user', text: 'next run' })
  await reopened.ready
  assert.equal((await stat(path)).ino, ino)
  const after = (await readFile(path, 'utf8')).trim().split('\n')
  assert.deepEqual([after.length, after.filter(line => line.includes('"ev":"trace"')).length, JSON.parse(after.at(-1)).text], [6, 1, 'next run'])
})
