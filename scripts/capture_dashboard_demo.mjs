#!/usr/bin/env node
// Capture the real local Dashboard with synthetic observations only.
import { spawn } from 'node:child_process'
import { createWriteStream } from 'node:fs'
import { copyFile, mkdtemp, mkdir, readdir, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import readline from 'node:readline'
import { chromium } from './dashboard-capture/node_modules/playwright/index.mjs'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const delay = ms => new Promise(resolve => setTimeout(resolve, ms))

function options() {
  const result = {
    output: path.join(root, 'docs/assets/dashboard'),
    python: path.join(root, '.venv/bin/python'),
    browser: process.env.DASHBOARD_DEMO_CHROMIUM,
    ffmpeg: process.env.DASHBOARD_DEMO_FFMPEG || 'ffmpeg',
  }
  const keys = { '--output-dir': 'output', '--python': 'python',
                 '--browser': 'browser', '--ffmpeg': 'ffmpeg' }
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = keys[process.argv[i]]
    if (!key || !process.argv[i + 1]) {
      throw new Error('Usage: node scripts/capture_dashboard_demo.mjs [--output-dir DIR] [--python PATH] [--browser PATH] [--ffmpeg PATH]')
    }
    result[key] = path.resolve(process.argv[i + 1])
  }
  return result
}

function control(child) {
  const messages = []
  const waiting = []
  const lines = readline.createInterface({ input: child.stdout })
  lines.on('line', line => {
    let message
    try { message = JSON.parse(line) }
    catch { message = { type: 'error', message: `invalid child response: ${line}` } }
    if (waiting.length) waiting.shift()(message)
    else messages.push(message)
  })
  return {
    async next() {
      const message = messages.length ? messages.shift() : await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error('demo process acknowledgment timed out')), 15000)
        waiting.push(value => { clearTimeout(timeout); resolve(value) })
      })
      if (message.type === 'error') throw new Error(message.message)
      return message
    },
    async send(step) {
      child.stdin.write(`${step}\n`)
      const message = await this.next()
      if (message.step !== step && !(step === 'stop' && message.type === 'stopped')) {
        throw new Error(`unexpected acknowledgment for ${step}: ${JSON.stringify(message)}`)
      }
      return message
    },
    close() { lines.close() },
  }
}

async function runEncoder(executable, frameDir, label, output, sourceFps) {
  const args = [
    '-hide_banner', '-loglevel', 'error', '-y',
    '-framerate', sourceFps.toFixed(4), '-i', path.join(frameDir, 'frame-%04d.png'),
    '-i', label,
    '-filter_complex',
    '[0:v]fps=8,scale=1200:-2:flags=lanczos,pad=1200:868:0:0:color=white[base];' +
    '[base][1:v]overlay=0:H-h,split[a][b];' +
    '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5',
    '-loop', '0', output,
  ]
  const proc = spawn(executable, args, { stdio: ['ignore', 'ignore', 'pipe'] })
  let stderr = ''
  proc.stderr.on('data', chunk => { stderr += chunk })
  const code = await new Promise((resolve, reject) => {
    proc.on('error', reject)
    proc.on('close', resolve)
  })
  if (code !== 0) throw new Error(`FFmpeg failed (${code}): ${stderr}`)
}

async function countGifFrames(executable, file) {
  const proc = spawn(executable, ['-hide_banner', '-i', file, '-f', 'null', '-'],
                     { stdio: ['ignore', 'ignore', 'pipe'] })
  let stderr = ''
  proc.stderr.on('data', chunk => { stderr += chunk })
  const code = await new Promise((resolve, reject) => {
    proc.on('error', reject)
    proc.on('close', resolve)
  })
  if (code !== 0) throw new Error(`GIF inspection failed: ${stderr}`)
  const matches = [...stderr.matchAll(/frame=\s*(\d+)/g)]
  return Number(matches.at(-1)?.[1] || 0)
}

async function capture() {
  const opt = options()
  const scratch = await mkdtemp(path.join(tmpdir(), 'nanobot-dashboard-capture-'))
  const frames = path.join(scratch, 'frames')
  await mkdir(frames)
  const child = spawn(opt.python, ['scripts/dashboard_demo.py', '--capture',
    '--data-dir', path.join(scratch, 'data')], {
    cwd: root, stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  })
  const childLog = createWriteStream(path.join(scratch, 'demo.stderr.log'))
  child.stderr.pipe(childLog)
  const ctl = control(child)
  let browser
  let success = false
  try {
    const ready = await ctl.next()
    if (ready.type !== 'ready') throw new Error(`unexpected demo response: ${JSON.stringify(ready)}`)
    const origin = new URL(ready.url).origin
    browser = await chromium.launch({ headless: true, executablePath: opt.browser,
                                      args: ['--no-sandbox'] })
    const context = await browser.newContext({
      viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1,
      locale: 'en-US', timezoneId: 'UTC', reducedMotion: 'reduce',
    })
    // Chromium requests a favicon that this Dashboard does not ship.
    await context.route('**/favicon.ico', route => route.fulfill({ status: 204 }))
    const page = await context.newPage()
    const errors = []
    const framesReceived = []
    page.on('console', message => { if (message.type() === 'error') errors.push(`console: ${message.text()}`) })
    page.on('pageerror', error => errors.push(`page: ${error.message}`))
    page.on('request', request => {
      if (/^https?:/.test(request.url()) && new URL(request.url()).origin !== origin) {
        errors.push(`unexpected nonlocal request: ${request.url()}`)
      }
    })
    page.on('requestfailed', request => {
      if (!request.failure()?.errorText.includes('ERR_ABORTED')) {
        errors.push(`failed request: ${request.url()} ${request.failure()?.errorText}`)
      }
    })
    page.on('response', response => {
      if (response.status() >= 400) errors.push(`HTTP ${response.status()}: ${response.url()}`)
    })
    page.on('websocket', socket => {
      if (new URL(socket.url()).origin.replace('ws:', 'http:') !== origin) {
        errors.push(`unexpected nonlocal WebSocket: ${socket.url()}`)
      }
      socket.on('framereceived', frame => {
        try { framesReceived.push(JSON.parse(frame.payload)) } catch { /* malformed frame is ignored by UI */ }
      })
    })
    const connected = page.getByText('Live updates connected', { exact: true })
    await page.goto(ready.url)
    await connected.waitFor()
    await page.getByRole('heading', { name: 'Overview' }).waitFor()
    await page.getByText('Runs per hour').waitFor()
    await page.getByRole('link', { name: `Open run ${ready.fixture_ids[0]}` }).waitFor()
    if (await page.locator('.recharts-bar-rectangle').count() === 0) {
      throw new Error('Overview hourly charts are empty')
    }
    await page.evaluate(() => document.fonts.ready)
    await delay(300)
    await page.screenshot({ path: path.join(scratch, 'overview.png') })

    await page.goto(`${ready.url}#/runs/${ready.fixture_ids[0]}`)
    await connected.waitFor()
    await page.locator('.run-detail h2', { hasText: ready.fixture_ids[0] }).waitFor()
    await page.locator('.run-detail summary', { hasText: 'kb_search' }).click()
    await page.getByText('Synthetic note: tests, docs, and dashboard assets are ready.').waitFor()
    await page.getByText('290 in / 90 out').waitFor()
    await page.getByText('Core records').waitFor()
    await page.locator('.run-detail summary', { hasText: 'Model step' }).first().click()
    await page.getByText('Synthetic answer: the checklist was reviewed.').waitFor()
    await page.evaluate(() => document.fonts.ready)
    await delay(250)
    await page.locator('.run-detail').screenshot({ path: path.join(scratch, 'runs.png') })

    await page.getByRole('link', { name: 'Runs', exact: true }).click()
    await page.getByText('3 LOADED').waitFor()
    await connected.waitFor()
    const baseline = await page.locator('.run-item').count()
    if (baseline !== 3) throw new Error(`expected three fixture runs, saw ${baseline}`)

    let recording = true
    let frameNumber = 0
    const recordingStarted = performance.now()
    const record = (async () => {
      while (recording) {
        await page.screenshot({ path: path.join(frames, `frame-${String(frameNumber++).padStart(4, '0')}.png`) })
        await delay(125)
      }
    })()
    try {
      await delay(2250)
      await ctl.send('start')
      await page.waitForFunction(id => location.hash === '#/runs' &&
        [...document.querySelectorAll('.run-item')].some(row =>
          row.getAttribute('href')?.endsWith(`/runs/${id}`) && row.textContent.includes('Running')),
      ready.live_id)
      if (!framesReceived.some(frame => frame.type === 'run.started' && frame.run_id === ready.live_id)) {
        throw new Error('run.started WebSocket frame was not received')
      }
      await delay(2200)
      await page.locator(`.run-item[href$="/runs/${ready.live_id}"]`).click()
      await page.locator('.run-detail h2', { hasText: ready.live_id }).waitFor()
      await delay(700)
      await ctl.send('memory')
      await page.getByText('Core records').waitFor()
      await delay(1100)
      await ctl.send('model')
      await page.getByText('Synthetic project: checking the release checklist.').waitFor()
      await delay(2400)
      await ctl.send('tools')
      await page.locator('.run-detail summary', { hasText: 'kb_search' }).waitFor()
      await delay(600)
      await page.locator('.run-detail summary', { hasText: 'kb_search' }).click()
      await page.getByText('Synthetic note: tests, docs, and dashboard assets are ready.').waitFor()
      await delay(3600)
      const finalAck = await ctl.send('final_model')
      if (finalAck.prompt_tokens !== 290 || finalAck.completion_tokens !== 90) {
        throw new Error('replay usage did not match model events')
      }
      await page.getByText('Synthetic answer: the checklist was reviewed.').waitFor()
      await page.locator('.run-detail summary', { hasText: 'kb_search' }).click()
      await page.getByText('Synthetic note: tests, docs, and dashboard assets are ready.').waitFor()
      await page.locator('.run-detail .trace-node').last().scrollIntoViewIfNeeded()
      await delay(1700)
      await page.evaluate(() => window.scrollTo(0, 0))
      await delay(900)
      await ctl.send('finish')
      await page.locator('.run-detail .status-completed').waitFor()
      await page.getByText('290 in / 90 out').waitFor()
      await page.locator('.run-detail summary', { hasText: 'kb_search' }).click()
      await page.getByText('Synthetic note: tests, docs, and dashboard assets are ready.').waitFor()
      const stored = await (await page.request.get(`${origin}/api/dashboard/runs/${ready.live_id}`)).json()
      if (stored.run.status !== 'completed' || stored.events.length !== 4 ||
          stored.run.prompt_tokens !== 290 || stored.run.completion_tokens !== 90) {
        throw new Error('terminal REST state is inconsistent')
      }
      if (!framesReceived.some(frame => frame.type === 'run.finished' && frame.run_id === ready.live_id)) {
        throw new Error('run.finished WebSocket frame was not received')
      }
      await delay(3200)
    } finally {
      recording = false
      await record
    }
    const recordingSeconds = (performance.now() - recordingStarted) / 1000
    if (errors.length) throw new Error(errors.join('\n'))
    await (async () => {
      const labelPage = await context.newPage({ viewport: { width: 1200, height: 34 } })
      await labelPage.setViewportSize({ width: 1200, height: 34 })
      await labelPage.setContent('<body style="margin:0;background:white;color:#52676a;font:12px Arial,sans-serif;display:flex;align-items:center;justify-content:flex-end;padding-right:16px;box-sizing:border-box;height:34px">Synthetic demo data</body>')
      await labelPage.screenshot({ path: path.join(scratch, 'label.png') })
      await labelPage.close()
    })()
    const count = (await readdir(frames)).length
    if (recordingSeconds < 15 || recordingSeconds > 25) {
      throw new Error(`recorded duration ${recordingSeconds.toFixed(2)}s is outside 15–25s`)
    }
    await runEncoder(opt.ffmpeg, frames, path.join(scratch, 'label.png'),
                     path.join(scratch, 'demo.gif'), count / recordingSeconds)
    const gifFrames = await countGifFrames(opt.ffmpeg, path.join(scratch, 'demo.gif'))
    if (Math.abs(gifFrames / 8 - recordingSeconds) > 1) {
      throw new Error(`GIF has ${gifFrames} frames; expected about ${Math.round(recordingSeconds * 8)}`)
    }
    await mkdir(opt.output, { recursive: true })
    for (const name of ['overview.png', 'runs.png', 'demo.gif']) {
      await copyFile(path.join(scratch, name), path.join(opt.output, name))
      console.log(`${path.join(opt.output, name)}: ${(await stat(path.join(opt.output, name))).size} bytes`)
    }
    console.log(`Captured ${count} frames over ${recordingSeconds.toFixed(2)} s; GIF has ${gifFrames} frames at 8 fps; no browser/request errors.`)
    success = true
  } finally {
    if (browser) await browser.close()
    if (child.exitCode === null) {
      try { await ctl.send('stop') } catch { /* EOF and process exit are also valid cleanup */ }
      child.stdin.end()
      await Promise.race([
        new Promise(resolve => child.once('exit', resolve)),
        delay(3000).then(() => child.kill('SIGTERM')),
      ])
    }
    ctl.close()
    childLog.end()
    if (success) await rm(scratch, { recursive: true, force: true })
    else console.error(`Capture diagnostics retained at ${scratch}`)
  }
}

capture().catch(error => { console.error(error); process.exitCode = 1 })
