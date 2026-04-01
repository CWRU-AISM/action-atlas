/**
 * Quick OpenVLA Screenshot Capture
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const CONFIG = {
  baseUrl: 'http://localhost:3000',
  outputDir: path.join(__dirname, '..', 'screenshots'),
  deviceScaleFactor: 3,
  viewport: { width: 1920, height: 1080 },
};

const TABS = [
  { id: 'features', label: 'Feature Explorer', waitFor: 'canvas, svg' },
  { id: 'videos', label: 'Demos', waitFor: 'video, img' },
  { id: 'ablation', label: 'Ablation Studies', waitFor: 'svg, canvas, .recharts-wrapper' },
  { id: 'pentest', label: 'Pen Testing', waitFor: 'video, img, canvas' },
  { id: 'findings', label: 'Findings', waitFor: 'svg, img' },
];

async function captureOpenVLA() {
  console.log('🚀 Starting OpenVLA screenshot capture...');

  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  });

  const page = await browser.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon')) {
      console.log(`   [ERROR] ${msg.text()}`);
    }
  });

  await page.setViewport({
    width: CONFIG.viewport.width,
    height: CONFIG.viewport.height,
    deviceScaleFactor: CONFIG.deviceScaleFactor,
  });

  try {
    console.log('\n📍 Loading application...');
    await page.goto(CONFIG.baseUrl, { waitUntil: 'networkidle0', timeout: 60000 });
    await wait(8000);

    // Switch to OpenVLA
    console.log('\n🔧 Switching to OpenVLA...');
    const selected = await page.evaluate(() => {
      const selects = document.querySelectorAll('select');
      for (const select of selects) {
        const options = Array.from(select.options);
        const hasModelOptions = options.some(opt => opt.value === 'pi05' || opt.value === 'openvla');
        if (hasModelOptions) {
          for (const opt of options) {
            if (opt.value === 'openvla') {
              select.value = 'openvla';
              select.dispatchEvent(new Event('change', { bubbles: true }));
              return opt.text;
            }
          }
        }
      }
      return null;
    });

    if (selected) {
      console.log(`   Selected: ${selected}`);
      await wait(8000);
    } else {
      console.log('   Could not switch to OpenVLA');
      await browser.close();
      return;
    }

    // Capture each tab
    for (const tab of TABS) {
      console.log(`\n📸 Capturing: ${tab.label}`);

      // Click tab
      const clicked = await page.evaluate((label) => {
        const buttons = Array.from(document.querySelectorAll('button, [role="tab"]'));
        const btn = buttons.find(b => b.textContent?.includes(label));
        if (btn) {
          btn.scrollIntoView({ block: 'center' });
          btn.click();
          return { found: true, text: btn.textContent?.trim() };
        }
        return { found: false, text: null };
      }, tab.label);

      if (!clicked.found) {
        console.log(`   ⚠️ Tab not found: ${tab.label}`);
        continue;
      }
      console.log(`   Clicked: "${clicked.text}"`);

      await wait(10000);

      // For Feature Explorer, zoom in and show points
      if (tab.id === 'features') {
        const svgElement = await page.$('svg');
        if (svgElement) {
          const box = await svgElement.boundingBox();
          if (box) {
            const centerX = box.x + box.width / 2;
            const centerY = box.y + box.height / 2;
            await page.mouse.move(centerX, centerY);
            for (let i = 0; i < 4; i++) {
              await page.mouse.wheel({ deltaY: -150 });
              await wait(300);
            }
          }
        }
        await wait(2000);

        // Click on a point
        await page.evaluate(() => {
          const points = document.querySelectorAll('circle:not([r="0"])');
          if (points.length > 0) {
            points[Math.floor(points.length / 2)]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          }
        });
        await wait(2000);
      }

      // Scroll to top
      await page.evaluate(() => {
        const main = document.querySelector('main') || document.body;
        main.scrollTop = 0;
      });
      await wait(1000);

      const filename = `action-atlas_${tab.id}_openvla_3x.png`;
      const filepath = path.join(CONFIG.outputDir, filename);
      await page.screenshot({ path: filepath, type: 'png', fullPage: false });

      const stats = fs.statSync(filepath);
      console.log(`   ✅ Saved: ${filename} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
    }

    console.log('\n✨ Complete!');
    console.log(`📁 ${CONFIG.outputDir}`);

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'error_openvla.png') });
  } finally {
    await browser.close();
  }
}

captureOpenVLA();
