const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({ headless: 'new', args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  page.on('pageerror', err => console.log('PAGE ERROR:', err.message.slice(0, 300)));
  page.on('console', msg => { if (msg.type() === 'error') console.log('CONSOLE ERROR:', msg.text().slice(0, 300)); });

  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto('http://localhost:3002', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 10000));

  // Try clicking Ablation tab using page.$$
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await btn.evaluate(el => el.textContent);
    if (text && text.includes('Ablation')) {
      console.log('Found Ablation button, clicking...');
      await btn.click();
      break;
    }
  }

  await new Promise(r => setTimeout(r, 5000));

  // Check what's showing
  const state = await page.evaluate(() => {
    const headers = Array.from(document.querySelectorAll('h1, h2, h3')).map(h => h.textContent.trim());
    const hasRecharts = !!document.querySelector('.recharts-wrapper');
    const mainContent = document.querySelector('.flex-1.overflow-hidden');
    const text = mainContent ? mainContent.textContent.slice(0, 300) : 'no main content';
    return { headers, hasRecharts, text };
  });
  console.log('After click state:', JSON.stringify(state, null, 2));

  await page.screenshot({ path: '/home/nvidia/robotsteering/vla_interp/vla_conceptviz/frontend/screenshots/test_prod_ablation.png' });
  await browser.close();
})();
