const { chromium } = require('playwright-core');

async function main() {
  const title = process.argv.slice(2).join(' ').trim();
  if (!title) throw new Error('Usage: node inspect-cnki-download.js <title>');
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
  const link = page.locator('a.fz14, a[href*="/article/abstract"]').filter({ hasText: title }).first();
  const href = await link.getAttribute('href');
  if (!href) throw new Error('No matching CNKI result.');
  await page.goto(new URL(href, page.url()).href, { waitUntil: 'domcontentloaded', timeout: 60000 });
  await page.waitForTimeout(2200);
  const links = await page.evaluate(() => [...document.querySelectorAll('a')].map((anchor) => ({
    text: String(anchor.innerText || anchor.title || '').replace(/\s+/g, ' ').trim(),
    href: anchor.href,
    className: anchor.className,
    target: anchor.target,
  })).filter((item) => /下载|阅读/.test(item.text)));
  process.stdout.write(`${JSON.stringify({ title: await page.title(), url: page.url(), links }, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
