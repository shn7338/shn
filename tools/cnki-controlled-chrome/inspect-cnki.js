const { chromium } = require('playwright-core');

async function main() {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const pages = browser.contexts().flatMap((context) => context.pages());
  const cnkiPages = pages.filter((page) => /cnki\.net/i.test(page.url()));
  if (!cnkiPages.length) throw new Error('No CNKI page is open.');

  let page = cnkiPages[0];
  for (const candidate of cnkiPages) {
    const text = await candidate.locator('body').innerText().catch(() => '');
    if (/北京交通大学/.test(text)) {
      page = candidate;
      break;
    }
  }

  const details = await page.evaluate(() => {
    const clean = (value) => String(value || '').replace(/\s+/g, ' ').trim();
    const attrs = (element) => ({
      tag: element.tagName.toLowerCase(),
      id: element.id || '',
      name: element.getAttribute('name') || '',
      type: element.getAttribute('type') || '',
      placeholder: element.getAttribute('placeholder') || '',
      ariaLabel: element.getAttribute('aria-label') || '',
      text: clean(element.innerText || element.value || '').slice(0, 160),
    });

    return {
      title: document.title,
      url: location.href,
      inputs: [...document.querySelectorAll('input, textarea, select')].map(attrs).slice(0, 80),
      buttons: [...document.querySelectorAll('button, [role="button"], input[type="submit"]')].map(attrs).slice(0, 80),
      links: [...document.querySelectorAll('a')].map((element) => ({
        text: clean(element.innerText).slice(0, 160),
        href: element.href,
      })).filter((item) => item.text).slice(0, 120),
      bodyPreview: clean(document.body?.innerText).slice(0, 4000),
    };
  });

  process.stdout.write(`${JSON.stringify(details, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.stack || error.message || error);
  process.exit(1);
});
