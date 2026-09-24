const { chromium } = require('playwright-core');

function clean(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

async function main() {
  const titles = process.argv.slice(2).map(clean).filter(Boolean);
  if (!titles.length) throw new Error('Usage: node detail-by-search-cnki.js <title> [title ...]');

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  let page = context.pages().find((candidate) => /cnki\.net/i.test(candidate.url())) || await context.newPage();
  const detailPage = await context.newPage();
  const output = [];

  for (const requestedTitle of titles) {
    await page.goto('https://www.cnki.net/', { waitUntil: 'domcontentloaded', timeout: 60000 });
    const input = page.locator('#txt_SearchText');
    await input.waitFor({ state: 'visible', timeout: 30000 });
    await input.fill(requestedTitle);
    await input.press('Enter');
    await page.waitForLoadState('domcontentloaded', { timeout: 60000 }).catch(() => null);
    await page.waitForTimeout(3000);

    const links = await page.evaluate(() => [...document.querySelectorAll('a.fz14, a[href*="/article/abstract"]')]
      .map((anchor) => ({ title: String(anchor.innerText || anchor.title || '').replace(/\s+/g, ' ').trim(), url: anchor.href }))
      .filter((item, index, all) => item.title && item.url && all.findIndex((other) => other.url === item.url) === index));
    const match = links.find((item) => item.title === requestedTitle) || links[0];
    if (!match) {
      output.push({ requestedTitle, error: 'No result found' });
      continue;
    }

    await detailPage.goto(match.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await detailPage.waitForTimeout(1800);
    const bodyText = clean(await detailPage.locator('body').innerText().catch(() => ''));
    const abstractIndex = bodyText.search(/摘要[:：]/);
    output.push({
      requestedTitle,
      title: match.title,
      url: detailPage.url(),
      excerpt: (abstractIndex >= 0 ? bodyText.slice(abstractIndex) : bodyText).slice(0, 5000),
    });
    await page.waitForTimeout(1200);
  }

  await detailPage.close();
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
