import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { registerHooks } from 'node:module'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'

// A file another program holds open, as Windows lets one hold it: the paths a test names in `rename` cannot be renamed
// over, and those it names in `open` cannot be opened to be written either. The trace module alone sees these.
globalThis.heldFiles = { rename: new Set(), open: new Set() }
const HELD = `
  import * as fs from 'node:fs/promises'
  export * from 'node:fs/promises'
  const held = globalThis.heldFiles
  const refused = path => Object.assign(new Error('EPERM: operation not permitted, ' + path), { code: 'EPERM' })
  export async function rename(from, to) {
    if (held.rename.has(String(to))) throw refused(to)
    return fs.rename(from, to)
  }
  export async function open(path, flags, mode) {
    if (flags === 'r+' && held.open.has(String(path))) throw refused(path)
    return fs.open(path, flags, mode)
  }
`
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === 'node:fs/promises' && context.parentURL?.endsWith('/src/trace.js')) return { url: `data:text/javascript,${encodeURIComponent(HELD)}`, shortCircuit: true }
    return nextResolve(specifier, context)
  },
})
const { createTrace } = await import('../src/trace.js')

// the own part of a synthetic key that a trace of an earlier version holds
const OWN = 'Qw7Er4Ty9Ui2Op5As8'

test('a trace an earlier version wrote that cannot be replaced is scrubbed in place, and one that cannot be scrubbed is never written to', async t => {
  const dir = await mkdtemp(join(tmpdir(), 's1-trace-held-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  const old = `${JSON.stringify({ t: '2026-11-01T00:00:00.000Z', ev: 'result', preview: `the key\nsk-vsr-${OWN}` })}\n`

  // held open: it cannot be replaced, so its scrubbed lines are written over it
  const held = join(dir, 'held.jsonl')
  await writeFile(held, old)
  globalThis.heldFiles.rename.add(held)
  const trace = createTrace(held)
  trace({ ev: 'user', text: 'hello' })
  await trace.ready
  const text = await readFile(held, 'utf8')
  const lines = text.trim().split('\n').map(line => JSON.parse(line))
  assert.deepEqual([lines[0].ev, lines[0].redaction, lines[1].ev, lines.at(-1).ev], ['trace', 1, 'result', 'user'])
  assert.ok(!text.includes(OWN), 'no part of the old key is left')

  // nor can it be written over: no record joins what it holds, and a warning says why
  const locked = join(dir, 'locked.jsonl')
  await writeFile(locked, old)
  globalThis.heldFiles.rename.add(locked)
  globalThis.heldFiles.open.add(locked)
  const warned = []
  const write = process.stderr.write
  process.stderr.write = text => warned.push(String(text)) > 0
  t.after(() => {
    process.stderr.write = write
  })
  const stopped = createTrace(locked)
  stopped({ ev: 'user', text: 'hello' })
  await stopped.ready
  stopped({ ev: 'user', text: 'again' })
  process.stderr.write = write
  assert.equal(await readFile(locked, 'utf8'), old)
  assert.match(warned.join(''), /^\[System One\] The trace is off: .*locked\.jsonl holds records an earlier version wrote, which may hold keys, and it could not be scrubbed \(EPERM\)\./)
})
