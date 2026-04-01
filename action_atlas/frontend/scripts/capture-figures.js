/**
 * Capture screenshots matching paper/figures/ filenames
 * Output to screenshots/figures/ with same names
 * MUST use production build (npm run build && npx next start -p 3002)
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const CONFIG = {
  baseUrl: 'http://localhost:3002',
  outputDir: path.join(__dirname, '..', 'screenshots', 'figures'),
  deviceScaleFactor: 3,
  viewport: { width: 1920, height: 1080 },
};

// Exact figures needed, matching paper/figures/ names
const CAPTURES = [
  // Pi0.5 captures
  { model: 'pi05', tab: 'Pen Testing', filename: 'action-atlas_pentest_3x.png' },
  { model: 'pi05', tab: 'Layer Wires', filename: 'action-atlas_wires_3x.png' },
  // OpenVLA captures
  { model: 'openvla', tab: 'Feature Explorer', filename: 'action-atlas_features_openvla_3x.png' },
  { model: 'openvla', tab: 'Ablation Studies', filename: 'action-atlas_ablation_openvla_3x.png' },
  { model: 'openvla', tab: 'Demos', filename: 'action-atlas_videos_openvla_3x.png' },
];

const MOCK_FEATURE = {
  data: {
    feature_info: {
      feature_id: '100',
      sae_id: 'action_expert_layer_12-concepts',
      layer: 12,
      type: 'action_expert',
      index: 100,
    },
    activation_data: [],
    explanation: 'Placing a bowl on top of a surface \u2014 encodes "put" action with bowl and "top" spatial relationship.',
    raw_stats: {
      neg_tokens: { tokens: [], values: [] },
      pos_tokens: { tokens: [], values: [] },
      freq_histogram: { heights: [], values: [] },
      logits_histogram: { heights: [], values: [] },
      similar_features: { feature_ids: [], values: [] },
    },
    bins_statistics: { bins_data: [] },
  },
};

async function clickTab(page, label) {
  const buttons = await page.$$('button');
  for (const btn of buttons) {
    const text = await btn.evaluate(el => el.textContent);
    if (text && text.includes(label)) {
      await btn.click();
      return text.trim();
    }
  }
  return null;
}

async function selectFeature(page) {
  return await page.evaluate((feature) => {
    const store = window.__REDUX_STORE__;
    if (!store || typeof store.dispatch !== 'function') return false;
    store.dispatch({ type: 'feature/setSelectedFeature', payload: feature });
    return true;
  }, MOCK_FEATURE);
}

async function switchModel(page, model) {
  if (model === 'pi05') return true; // default

  for (let attempt = 1; attempt <= 3; attempt++) {
    const selected = await page.evaluate((targetModel) => {
      const selects = document.querySelectorAll('select');
      for (const select of selects) {
        const options = Array.from(select.options);
        if (options.some(opt => opt.value === 'pi05' || opt.value === 'openvla')) {
          for (const opt of options) {
            if (opt.value === targetModel) {
              select.value = targetModel;
              select.dispatchEvent(new Event('change', { bubbles: true }));
              return opt.text;
            }
          }
        }
      }
      return null;
    }, model);

    if (selected) {
      console.log(`   Model: ${selected}`);
      await wait(10000);
      // Dismiss errors
      await page.evaluate(() => {
        document.querySelectorAll('[aria-label="close"], .MuiAlert-action button, .close-button')
          .forEach(btn => btn.click());
      });
      await wait(1000);
      return true;
    }
    await wait(3000);
  }
  return false;
}

async function captureScreenshots() {
  console.log('Capturing paper figures...\n');

  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-web-security'],
  });

  const page = await browser.newPage();
  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon') && !msg.text().includes('404'))
      console.log(`   [ERROR] ${msg.text().slice(0, 120)}`);
  });
  page.on('pageerror', err => console.log(`   [PAGE ERROR] ${err.message.slice(0, 120)}`));

  await page.setViewport({
    width: CONFIG.viewport.width,
    height: CONFIG.viewport.height,
    deviceScaleFactor: CONFIG.deviceScaleFactor,
  });

  try {
    console.log('Loading app...');
    await page.goto(CONFIG.baseUrl, { waitUntil: 'networkidle0', timeout: 60000 });
    await wait(12000);

    // Set feature for steering panel
    await selectFeature(page);
    await wait(2000);

    let currentModel = 'pi05';

    for (const cap of CAPTURES) {
      console.log(`\n--- ${cap.filename} ---`);

      // Switch model if needed
      if (cap.model !== currentModel) {
        // Go to safe tab first
        await clickTab(page, 'Demos');
        await wait(2000);

        if (!await switchModel(page, cap.model)) {
          console.log(`   SKIP: could not switch to ${cap.model}`);
          continue;
        }
        currentModel = cap.model;

        // Re-inject feature after model switch
        await selectFeature(page);
        await wait(2000);
      }

      // Click tab
      const clicked = await clickTab(page, cap.tab);
      if (!clicked) {
        console.log(`   SKIP: tab "${cap.tab}" not found`);
        await page.screenshot({ path: path.join(CONFIG.outputDir, `debug_${cap.filename}`) });
        continue;
      }
      console.log(`   Tab: ${clicked}`);

      await wait(10000);

      // For Feature Explorer, try to interact with scatter if possible
      if (cap.tab === 'Feature Explorer') {
        // Try clicking a point (may not work in headless but worth trying)
        await page.evaluate(() => {
          const points = document.querySelectorAll('circle:not([r="0"]), .scatter-point');
          if (points.length > 0) {
            points[Math.floor(points.length / 2)]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
          }
        });
        await wait(3000);
      }

      // Scroll to top
      await page.evaluate(() => {
        const main = document.querySelector('main') || document.body;
        main.scrollTop = 0;
      });
      await wait(1000);

      const filepath = path.join(CONFIG.outputDir, cap.filename);
      await page.screenshot({ path: filepath, type: 'png', fullPage: false });

      const stats = fs.statSync(filepath);
      console.log(`   Saved: ${cap.filename} (${(stats.size / 1024).toFixed(0)} KB)`);
    }

    console.log('\n\nDone!');
    console.log(`Output: ${CONFIG.outputDir}`);

  } catch (error) {
    console.error('\nError:', error.message);
    try { await page.screenshot({ path: path.join(CONFIG.outputDir, 'error.png') }); } catch (e) {}
  } finally {
    await browser.close();
  }
}

captureScreenshots();
