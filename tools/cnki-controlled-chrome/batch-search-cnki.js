const { chromium } = require('playwright-core');

function clean(value) {
  return String(value || '').replace(/\s+/g, ' ').trim();
}

async function parseResults(page) {
  return page.evaluate(() => {
    const cleanText = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const rows = [...document.querySelectorAll('.result-table-list tbody tr, table tbody tr')];
    return rows.slice(0, 12).map((row) => {
      const titleLink = row.querySelector('a.fz14, a[href*="/article/abstract"]');
      const downloadLink = row.querySelector('a.downloadlink, a.icon-download');
      return {
        text: cleanText(row.innerText).slice(0, 800),
        title: cleanText(titleLink?.innerText || titleLink?.title),
        detailUrl: titleLink?.href || '',
        downloadUrl: downloadLink?.href || '',
      };
    }).filter((item) => item.title);
  });
}

async function main() {
  const queries = process.argv.slice(2).map(clean).filter(Boolean);
  if (!queries.length) throw new Error('Usage: node batch-search-cnki.js <query> [query ...]');

  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const context = browser.contexts()[0];
  let page = context.pages().find((candidate) => /cnki\.net/i.test(candidate.url()));
  if (!page) page = await context.newPage();
  await page.bringToFront();

  const output = [];
  for (const query of queries) {
    await page.goto('https://www.cnki.net/', { waitUntil: 'domcontentloaded', timeout: 60000 });
    const input = page.locator('#txt_SearchText');
    await input.waitFor({ state: 'visible', timeout: 30000 });
    await input.fill(query);
    await input.press('Enter');
    await page.waitForLoadState('domcontentloaded', { timeout: 60000 }).catch(() => null);
    await page.waitForTimeout(3500);

    const body = await page.locator('body').innerText().catch(() => '');
    output.push({
      query,
      url: page.url(),
      securityVerification: /拖动下方拼图完成验证|安全验证/.test(body),
      results: await parseResults(page),
    });
    await page.waitForTimeout(1800);
  }

  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
