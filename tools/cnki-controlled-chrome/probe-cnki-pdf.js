const { chromium } = require('playwright-core');

async function main() {
  const title = process.argv.slice(2).join(' ').trim();
  if (!title) throw new Error('Usage: node probe-cnki-pdf.js <title>');
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  let page = context.pages().find((candidate) => /cnki\.net/i.test(candidate.url())) || await context.newPage();
  await page.goto('https://www.cnki.net/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  const input = page.locator('#txt_SearchText');
  await input.waitFor({ state: 'visible', timeout: 30000 });
  await input.fill(title);
  await input.press('Enter');
  await page.waitForLoadState('domcontentloaded', { timeout: 60000 }).catch(() => null);
  await page.waitForTimeout(3000);
  const titleLink = page.locator('a.fz14, a[href*="/article/abstract"]').filter({ hasText: title }).first();
  const detailUrl = await titleLink.getAttribute('href');
  if (!detailUrl) throw new Error('No matching result.');
  await page.goto(new URL(detailUrl, page.url()).href, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2200);
  const pdfUrl = await page.getByText('PDF下载', { exact: true }).first().getAttribute('href');
  if (!pdfUrl) throw new Error('PDF download link is unavailable.');
  const response = await context.request.get(new URL(pdfUrl, page.url()).href, {
    headers: { referer: page.url() },
    timeout: 120000,
  });
  const body = await response.body();
  process.stdout.write(`${JSON.stringify({
    status: response.status(),
    url: response.url(),
    contentType: response.headers()['content-type'] || '',
    contentDisposition: response.headers()['content-disposition'] || '',
    size: body.length,
    signatureHex: body.subarray(0, 16).toString('hex'),
    textPreview: /^text\//i.test(response.headers()['content-type'] || '') ? body.toString('utf8', 0, 500) : '',
  }, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
