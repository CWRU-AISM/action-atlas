/**
 * Targeted Screenshot Capture - Ablation, Videos, Findings, Pen Testing
 * Captures specific tabs for both PI05 and OpenVLA at 3x (600+ DPI for print)
 * MUST use production build (npm run build && npx next start -p 3002)
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const CONFIG = {
  baseUrl: 'http://localhost:3002',
  outputDir: path.join(__dirname, '..', 'screenshots', '6x'),
  deviceScaleFactor: 3,
  viewport: { width: 1920, height: 1080 },
};

const TABS = [
  { id: 'videos', label: 'Demos', waitFor: 'video, img' },
  { id: 'ablation', label: 'Ablation Studies', waitFor: '.recharts-wrapper' },
  { id: 'pentest', label: 'Pen Testing', waitFor: 'video, img, canvas' },
  { id: 'findings', label: 'Findings', waitFor: 'svg, img' },
];

const MODELS = ['pi05', 'openvla'];

// Mock feature data to populate the Feature Steering panel
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
    explanation: 'Placing a bowl on top of a surface — encodes "put" action with bowl and "top" spatial relationship.',
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

async function selectFeature(page, featureData) {
  // Dispatch Redux action via window.__REDUX_STORE__ (exposed in provider.tsx)
  return await page.evaluate((feature) => {
    const store = window.__REDUX_STORE__;
    if (!store || typeof store.dispatch !== 'function') {
      console.error('Redux store not found on window.__REDUX_STORE__');
      return false;
    }
    store.dispatch({
      type: 'feature/setSelectedFeature',
      payload: feature
    });
    return true;
  }, featureData);
}

async function captureScreenshots() {
  console.log('Starting targeted screenshot capture (production mode)...');

  if (!fs.existsSync(CONFIG.outputDir)) {
    fs.mkdirSync(CONFIG.outputDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-web-security',
    ],
  });

  const page = await browser.newPage();

  page.on('console', msg => {
    if (msg.type() === 'error' && !msg.text().includes('favicon') && !msg.text().includes('404')) {
      console.log(`   [ERROR] ${msg.text().slice(0, 150)}`);
    }
  });

  page.on('pageerror', err => {
    console.log(`   [PAGE ERROR] ${err.message.slice(0, 150)}`);
  });

  await page.setViewport({
    width: CONFIG.viewport.width,
    height: CONFIG.viewport.height,
    deviceScaleFactor: CONFIG.deviceScaleFactor,
  });

  try {
    console.log('\nLoading application...');
    await page.goto(CONFIG.baseUrl, {
      waitUntil: 'networkidle0',
      timeout: 60000
    });

    console.log('   Waiting for hydration...');
    await wait(12000);

    // Select a feature to populate the Feature Steering panel
    console.log('   Selecting feature for steering panel...');
    const featureSet = await selectFeature(page, MOCK_FEATURE);
    console.log(`   Feature selection: ${featureSet ? 'OK (via Redux)' : 'trying fallback...'}`);
    await wait(2000);

    // Verify steering panel is populated
    const steeringVisible = await page.evaluate(() => {
      const text = document.body.textContent;
      return text.includes('Steering Magnitude') || text.includes('APPLY STEERING');
    });
    console.log(`   Steering panel visible: ${steeringVisible}`);

    for (const model of MODELS) {
      console.log(`\n=== Model: ${model.toUpperCase()} ===`);

      if (model !== 'pi05') {
        // Navigate to Demos tab first (avoid feature explorer canvas issues)
        await clickTab(page, 'Demos');
        await wait(2000);

        // Switch model via select dropdown
        let selected = null;
        for (let attempt = 1; attempt <= 3; attempt++) {
          console.log(`   Model switch attempt ${attempt}...`);

          selected = await page.evaluate((targetModel) => {
            const selects = document.querySelectorAll('select');
            for (const select of selects) {
              const options = Array.from(select.options);
              const hasModelOptions = options.some(opt =>
                opt.value === 'pi05' || opt.value === 'openvla'
              );
              if (hasModelOptions) {
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

          if (selected) break;
          await wait(3000);
        }

        if (selected) {
          console.log(`   Selected: ${selected}`);
          await wait(10000);

          // Re-inject feature selection after model switch
          await selectFeature(page, MOCK_FEATURE);
          await wait(2000);

          // Dismiss any error overlays
          await page.evaluate(() => {
            const closeButtons = document.querySelectorAll('[aria-label="close"], .MuiAlert-action button, .close-button');
            closeButtons.forEach(btn => btn.click());
          });
          await wait(1000);
        } else {
          console.log(`   Could not switch to ${model}, skipping...`);
          continue;
        }
      }

      for (const tab of TABS) {
        console.log(`\n   Capturing: ${tab.label} (${model})`);

        const clickedText = await clickTab(page, tab.label);

        if (!clickedText) {
          console.log(`   Tab not found: ${tab.label}`);
          await page.screenshot({ path: path.join(CONFIG.outputDir, `debug_${model}_${tab.id}.png`) });
          continue;
        }
        console.log(`   Clicked: "${clickedText}"`);

        await wait(8000);

        if (tab.waitFor) {
          try {
            await page.waitForSelector(tab.waitFor, { timeout: 8000 });
            console.log(`   Content ready`);
          } catch (e) {
            console.log(`   Content not found: ${tab.waitFor} (capturing anyway)`);
          }
        }

        await page.evaluate(() => {
          const main = document.querySelector('main') || document.body;
          main.scrollTop = 0;
        });
        await wait(1000);

        const filename = model === 'pi05'
          ? `action-atlas_${tab.id}_6x.png`
          : `action-atlas_${tab.id}_${model}_6x.png`;
        const filepath = path.join(CONFIG.outputDir, filename);

        await page.screenshot({ path: filepath, type: 'png', fullPage: false });

        const stats = fs.statSync(filepath);
        console.log(`   Saved: ${filename} (${(stats.size / 1024).toFixed(0)} KB)`);
      }
    }

    console.log('\n\nDone! All targeted screenshots captured.');
    console.log(`Output: ${CONFIG.outputDir}`);

  } catch (error) {
    console.error('\nError:', error.message);
    try {
      await page.screenshot({ path: path.join(CONFIG.outputDir, 'error_targeted.png') });
    } catch (e) {}
  } finally {
    await browser.close();
  }
}

captureScreenshots();
