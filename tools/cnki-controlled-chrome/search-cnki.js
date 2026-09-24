const { chromium } = require('playwright-core');

async function chooseCnkiPage(browser) {
  const pages = browser.contexts().flatMap((context) => context.pages());
  const candidates = pages.filter((page) => /cnki\.net/i.test(page.url()));
  if (!candidates.length) throw new Error('No CNKI page is open.');
  for (const page of candidates) {
    const text = await page.locator('body').innerText().catch(() => '');
    if (/北京交通大学/.test(text)) return page;
  }
  return candidates[0];
}

async function main() {
  const query = process.argv.slice(2).join(' ').trim();
  if (!query) throw new Error('Usage: node search-cnki.js <query>');

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const page = await chooseCnkiPage(browser);
  await page.bringToFront();

  if (!/^https:\/\/www\.cnki\.net\/?(?:#.*)?$/i.test(page.url())) {
    await page.goto('https://www.cnki.net/', { waitUntil: 'domcontentloaded', timeout: 60000 });
  }

  const input = page.locator('#txt_SearchText');
  await input.waitFor({ state: 'visible', timeout: 30000 });
  await input.fill(query);

  await Promise.all([
    page.waitForLoadState('domcontentloaded', { timeout: 60000 }).catch(() => null),
    input.press('Enter'),
  ]);

  await page.waitForTimeout(5000);
  const bodyText = await page.locator('body').innerText().catch(() => '');
  process.stdout.write(`${JSON.stringify({
    query,
    title: await page.title().catch(() => ''),
    url: page.url(),
    institutionVisible: /北京交通大学/.test(bodyText),
    securityVerification: /拖动下方拼图完成验证|安全验证/.test(bodyText),
    bodyPreview: bodyText.replace(/\s+/g, ' ').slice(0, 8000),
  }, null, 2)}\n`);

  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
