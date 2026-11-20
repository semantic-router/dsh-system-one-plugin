import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { test } from 'node:test'
import { personalShape, pseudonym, pseudonymKeys } from '../src/pseudonyms.js'

test('an id shaped as an IP address is one of version 4 or 6, and an email address may have its domain in any script', () => {
  assert.equal(personalShape('2001:db8::1'), 'IP_ADDRESS')
  assert.equal(personalShape('[2001:db8::1]'), 'IP_ADDRESS')
  assert.equal(personalShape('203.0.113.7'), 'IP_ADDRESS')
  assert.equal(personalShape('999.999.999.999'), undefined, 'four numbers that are no bytes are no address')
  assert.equal(personalShape('ada@例え.テスト'), 'EMAIL_ADDRESS')
  assert.equal(personalShape('ada@xn--r8jz45g.xn--zckzah'), 'EMAIL_ADDRESS')
  assert.equal(personalShape('ada@bücher.example'), 'EMAIL_ADDRESS')
  assert.equal(personalShape('+1 415 555 0134'), 'PHONE_NUMBER')
  assert.equal(personalShape('2026-11-20'), undefined)
  assert.equal(personalShape('88231'), undefined)
})

test('two first uses of one key file at once share one key, and a key file left empty is replaced by a key both keep', async t => {
  const dir = await mkdtemp(join(tmpdir(), 's1-pseudonyms-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  for (let trial = 0; trial < 20; trial++) {
    const path = join(dir, `race-${trial}`, 'pseudonyms.key')
    const [a, b] = await Promise.all([pseudonymKeys(path)(), pseudonymKeys(path)()])
    assert.equal(pseudonym(a, 'EMAIL_ADDRESS', 'ada@example.test'), pseudonym(b, 'EMAIL_ADDRESS', 'ada@example.test'), `trial ${trial}`)
  }
  const empty = join(dir, 'empty.key')
  await writeFile(empty, '')
  const first = await pseudonymKeys(empty)()
  const second = await pseudonymKeys(empty)()
  assert.ok(first.equals(second))
  assert.match(await readFile(empty, 'utf8'), /^[0-9a-f]{64}\n$/)
})
