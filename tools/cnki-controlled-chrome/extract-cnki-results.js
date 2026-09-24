const { chromium } = require('playwright-core');

async function main() {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const pages = browser.contexts().flatMap((context) => context.pages());
  const page = [...pages].reverse().find((candidate) => /kns\.cnki\.net\/kns8s\/defaultresult/i.test(candidate.url()));
  if (!page) throw new Error('No CNKI result page is open.');

  const data = await page.evaluate(() => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const selectors = ['.result-table-list tbody tr', 'table tbody tr', '.result-table-list tr', '.list-item'];
    let rows = [];
    for (const selector of selectors) {
      rows = [...document.querySelectorAll(selector)];
      if (rows.length) break;
    }

    return rows.slice(0, 50).map((row, index) => ({
      index: index + 1,
      text: clean(row.innerText).slice(0, 1000),
      links: [...row.querySelectorAll('a')].map((anchor) => ({
        text: clean(anchor.innerText || anchor.title),
        href: anchor.href,
        className: anchor.className,
      })).filter((item) => item.text || item.href),
    }));
  });

  process.stdout.write(`${JSON.stringify({ title: await page.title(), url: page.url(), rows: data }, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
