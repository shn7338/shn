const { chromium } = require('playwright-core');

async function main() {
  const browser = await chromium.connectOverCDP('http://127.0.0.1:9222');
  const pages = browser.contexts().flatMap((context) => context.pages());

  const summary = [];
  for (const page of pages) {
    const bodyText = await page.locator('body').innerText().catch(() => '');
    summary.push({
      title: await page.title().catch(() => ''),
      url: page.url(),
      cnkiPage: /cnki\.net|中国知网/i.test(`${page.url()} ${await page.title().catch(() => '')}`),
      institutionVisible: /北京交通大学/.test(bodyText),
      logoutVisible: /退出|注销/.test(bodyText),
      loginVisible: /登录|机构登录/.test(bodyText),
    });
  }

  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  await browser.close();
}

main().catch((error) => {
  console.error(error.message || error);
  process.exit(1);
});
