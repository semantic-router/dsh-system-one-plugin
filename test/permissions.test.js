import assert from 'node:assert/strict'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { basename, join } from 'node:path'
import { test } from 'node:test'
import { deniedText, mayWrite, writePolicy } from '../src/permissions.js'

// DeepSeek Harness's services as its base composition mounts them: the sandboxed filesystem and the policy home
const services = resolved => {
  const asked = []
  return { asked, sandboxPolicy: { resolve: request => (asked.push(request), typeof resolved === 'function' ? resolved() : resolved) }, fs: { sandboxMode: 'workspace-write' } }
}

test('a session\'s write policy is the one DeepSeek Harness applies to its own write tool', () => {
  const session = { id: 's', header: { cwd: '/work' } }
  const confined = services({ mode: 'workspace-write', workspaceRoot: '/work', additionalWritableRoots: ['/data/out'] })
  assert.deepEqual(writePolicy(confined, session), { mode: 'workspace-write', workspace: '/work', roots: ['/work', '/data/out', '/tmp', tmpdir()] })
  assert.deepEqual(confined.asked, [{ session }], 'resolved for the calling session')
  assert.deepEqual(writePolicy(services({ mode: 'read-only', workspaceRoot: '/work' }), session), { mode: 'read-only', roots: [] })
  assert.equal(writePolicy(services({ mode: 'danger-full-access', workspaceRoot: '/work' }), session), undefined)
  // a filesystem that confines nothing, or no policy home, writes anywhere, as the harness's write tool does there
  assert.equal(writePolicy({ ...services({ mode: 'read-only', workspaceRoot: '/work' }), fs: {} }, session), undefined)
  assert.equal(writePolicy({ fs: { sandboxMode: 'read-only' } }, session), undefined)
  assert.equal(writePolicy(undefined, session), undefined)
  // a policy that cannot be read, or a mode this plugin does not know, writes nothing
  assert.deepEqual(writePolicy(services(() => { throw new Error('the session working directory must be absolute') }), session), { mode: 'read-only', roots: [], unknown: 'the session working directory must be absolute' })
  assert.deepEqual(writePolicy(services({ mode: 'workspace-only', workspaceRoot: '/work' }), session), { mode: 'read-only', roots: [] })
})

test('a write lands where its real path is: inside the workspace yes, outside it, through a link or by .. no', async t => {
  const workspace = await mkdtemp(join(tmpdir(), 's1-ws-'))
  const outside = await mkdtemp(join(tmpdir(), 's1-out-'))
  t.after(async () => {
    await rm(workspace, { recursive: true, force: true })
    await rm(outside, { recursive: true, force: true })
  })
  const policy = { mode: 'workspace-write', workspace, roots: [workspace] }
  assert.equal(await mayWrite(policy, join(workspace, 'results', 'new', 'rows.jsonl')), true)
  assert.equal(await mayWrite(policy, workspace), true)
  assert.equal(await mayWrite(policy, join(outside, 'rows.jsonl')), false)
  assert.equal(await mayWrite(policy, join(workspace, '..', basename(outside), 'rows.jsonl')), false)
  assert.equal(await mayWrite({ mode: 'read-only', roots: [] }, join(workspace, 'rows.jsonl')), false)
  assert.equal(await mayWrite(undefined, join(outside, 'rows.jsonl')), true)

  if (process.platform !== 'win32') {
    await writeFile(join(outside, 'notes.txt'), 'x')
    await symlink(outside, join(workspace, 'elsewhere'))
    await symlink(join(outside, 'notes.txt'), join(workspace, 'notes.txt'))
    await symlink(join(outside, 'not-yet.txt'), join(workspace, 'dangling.txt'))
    await mkdir(join(workspace, 'real'))
    await symlink(join(workspace, 'real'), join(workspace, 'alias'))
    assert.equal(await mayWrite(policy, join(workspace, 'elsewhere', 'rows.jsonl')), false, 'a folder that links out')
    assert.equal(await mayWrite(policy, join(workspace, 'notes.txt')), false, 'a file that links out')
    assert.equal(await mayWrite(policy, join(workspace, 'dangling.txt')), false, 'a link to nothing yet, which a write would follow')
    assert.equal(await mayWrite(policy, join(workspace, 'alias', 'rows.jsonl')), true, 'a link that stays inside')
  }
})

test('a refused write says why, and where a write may go', () => {
  assert.equal(deniedText({ mode: 'workspace-write', workspace: '/work', roots: ['/work'] }, 'output', '/data/rows.jsonl'),
    'output /data/rows.jsonl is outside this session\'s workspace (/work), where DeepSeek Harness\'s file policy (workspace-write) lets tools write: name a path inside it, or ask the user to allow writing there')
  assert.equal(deniedText({ mode: 'read-only', roots: [] }, 'annotate', 'a.png'), 'annotate a.png cannot be written: this session\'s file policy is read-only')
})
