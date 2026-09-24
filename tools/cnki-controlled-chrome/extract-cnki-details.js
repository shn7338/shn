const { chromium } = require('playwright-core');

function clean(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

async function main() {
  const requestedTitles = process.argv.slice(2).map(clean).filter(Boolean);
  if (!requestedTitles.length) throw new Error('Usage: node extract-cnki-details.js <title> [title ...]');

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  const pages = context.pages();
  const resultPage = [...pages].reverse().find((page) => /kns\.cnki\.net\/kns8s\/defaultresult/i.test(page.url()));
  if (!resultPage) throw new Error('No CNKI result page is open.');

  const resultLinks = await resultPage.evaluate(() => [...document.querySelectorAll('a.fz14, a[href*="/article/abstract"]')]
    .map((anchor) => ({ title: String(anchor.innerText || anchor.title || '').replace(/\s+/g, ' ').trim(), url: anchor.href }))
    .filter((item, index, all) => item.title && item.url && all.findIndex((other) => other.url === item.url) === index));

  const selected = requestedTitles.map((title) => {
    const exact = resultLinks.find((item) => item.title === title);
    return exact || resultLinks.find((item) => item.title.includes(title) || title.includes(item.title));
  }).filter(Boolean);

  const output = [];
  const detailPage = await context.newPage();
  for (const item of selected) {
    await detailPage.goto(item.url, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await detailPage.waitForTimeout(2200);
    const bodyText = clean(await detailPage.locator('body').innerText().catch(() => ''));
    const abstractIndex = bodyText.search(/摘要[:：]/);
    const excerpt = abstractIndex >= 0 ? bodyText.slice(abstractIndex, abstractIndex + 5000) : bodyText.slice(0, 5000);
    output.push({
      title: item.title,
      url: detailPage.url(),
      pageTitle: await detailPage.title().catch(() => ''),
      excerpt,
    });
    await detailPage.waitForTimeout(1200);
  }
  await detailPage.close();

  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
