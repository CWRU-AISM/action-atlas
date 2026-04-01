/**
 * High-Resolution Screenshot Capture for Action Atlas
 * Captures all tabs for both PI05 and OpenVLA models
 * Enhanced: Multiple color modes and layer views for Feature Explorer
 */

const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const CONFIG = {
  baseUrl: 'http://localhost:3000',
  outputDir: path.join(__dirname, '..', 'screenshots'),
  deviceScaleFactor: 3,  // 3x gives 5760×3240 = 800+ DPI at full page width
  viewport: { width: 1920, height: 1080 },
};

// Color modes for Feature Explorer screenshots
const COLOR_MODES = ['cluster', 'concept', 'layer'];

// Layers to capture for Feature Explorer (subset for representative views)
// 'all' means the all_layers aggregate view
const FEATURE_EXPLORER_LAYERS = {
  pi05: ['all', 0, 5, 10, 15, 17],   // All layers + Early, mid, late
  openvla: ['all', 0, 8, 16, 24, 31], // All layers + Early, mid, late (32 total)
};

const TABS = [
  { id: 'features', label: 'Feature Explorer', waitFor: 'canvas, svg', needsClick: true, multiCapture: true },
  { id: 'wires', label: 'Layer Wires', waitFor: 'svg, canvas', skipModels: ['openvla'] },  // No SAE for OpenVLA yet
  { id: 'videos', label: 'Demos', waitFor: 'video, img' },
  { id: 'ablation', label: 'Ablation Studies', waitFor: 'svg, canvas, .recharts-wrapper' },
  { id: 'pentest', label: 'Pen Testing', waitFor: 'video, img, canvas' },
  { id: 'findings', label: 'Findings', waitFor: 'svg, img' },
];

const MODELS = ['pi05', 'openvla'];

async function captureScreenshots() {
  console.log('🚀 Starting screenshot capture for both models...');

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

  // Only log errors, not all failures
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
    await page.goto(CONFIG.baseUrl, {
      waitUntil: 'networkidle0',
      timeout: 60000
    });

    console.log('   Initial page loaded, waiting for hydration...');
    await wait(8000);

    // Capture for each model
    for (const model of MODELS) {
      console.log(`\n🔧 Switching to model: ${model.toUpperCase()}`);

      // For non-default models, switch model without reloading (reload causes errors)
      if (model !== 'pi05') {
        // First go back to Feature Explorer tab to reset state
        await page.evaluate(() => {
          const buttons = Array.from(document.querySelectorAll('button'));
          const btn = buttons.find(b => b.textContent?.includes('Feature Explorer'));
          if (btn) btn.click();
        });
        await wait(2000);

        // Try to switch model with retries
        let selected = null;
        for (let attempt = 1; attempt <= 3; attempt++) {
          console.log(`   Model switch attempt ${attempt}...`);

          // Use native select element to change model
          selected = await page.evaluate((targetModel) => {
            const selects = document.querySelectorAll('select');
            for (const select of selects) {
              // Find the select that has model options
              const options = Array.from(select.options);
              const hasModelOptions = options.some(opt =>
                opt.value === 'pi05' || opt.value === 'openvla'
              );
              if (hasModelOptions) {
                // Find and select the target model
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
          console.log(`   Selected model: ${selected}`);
          await wait(8000); // Wait for data to reload

          // Dismiss any error modals/alerts that might appear
          await page.evaluate(() => {
            const closeButtons = document.querySelectorAll('[aria-label="close"], .MuiAlert-action button, .close-button');
            closeButtons.forEach(btn => btn.click());
          });
          await wait(1000);

          // Check if tabs are still visible
          const tabsVisible = await page.evaluate(() => {
            const buttons = Array.from(document.querySelectorAll('button'));
            return buttons.filter(b =>
              b.textContent?.includes('Feature Explorer') ||
              b.textContent?.includes('Ablation') ||
              b.textContent?.includes('Demos')
            ).length;
          });
          console.log(`   Found ${tabsVisible} navigation tabs`);
        } else {
          console.log(`   Could not switch to ${model}, skipping...`);
          continue;
        }
      }

      await wait(500);

      for (const tab of TABS) {
        // Skip tabs that don't work for certain models
        if (tab.skipModels && tab.skipModels.includes(model)) {
          console.log(`\n⏭️  Skipping: ${tab.label} (${model}) - not supported`);
          continue;
        }

        console.log(`\n📸 Capturing: ${tab.label} (${model})`);

        // First dismiss any error overlays
        await page.evaluate(() => {
          const closeButtons = document.querySelectorAll('[aria-label="close"], .MuiAlert-action button, .MuiSnackbar button, .close-button, [data-dismiss]');
          closeButtons.forEach(btn => btn.click());
          // Also press escape to close modals
          document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        });
        await wait(300);

        // Click the tab with better matching
        const clicked = await page.evaluate((label) => {
          // Try various selectors for tab buttons
          const allButtons = Array.from(document.querySelectorAll('button, [role="tab"]'));
          // Exact partial match
          let btn = allButtons.find(b => b.textContent?.includes(label));
          // Try shorter match for abbreviated labels
          if (!btn && label.length > 6) {
            const shortLabel = label.substring(0, 6);
            btn = allButtons.find(b => b.textContent?.includes(shortLabel));
          }
          if (btn) {
            // Scroll into view first
            btn.scrollIntoView({ block: 'center' });
            btn.click();
            return { found: true, text: btn.textContent?.trim() };
          }
          return { found: false, text: null };
        }, tab.label);

        if (!clicked.found) {
          console.log(`   ⚠️ Tab not found: ${tab.label}`);
          // Take a debug screenshot
          await page.screenshot({ path: path.join(CONFIG.outputDir, `debug_${model}_${tab.id}.png`) });
          continue;
        }
        console.log(`   Clicked: "${clicked.text}"`);

        // Wait for content to load
        console.log(`   Waiting for content...`);
        await wait(10000);  // 10 seconds per tab for full data loading

        // Try to wait for specific content
        if (tab.waitFor) {
          try {
            await page.waitForSelector(tab.waitFor, { timeout: 5000 });
            console.log(`   Found expected content: ${tab.waitFor}`);
          } catch (e) {
            console.log(`   Content selector not found: ${tab.waitFor}`);
          }
        }

        // For Feature Explorer - enhanced capture with zoom, color modes, and layers
        if (tab.id === 'features' && tab.multiCapture) {
          console.log('   📊 Starting enhanced Feature Explorer capture...');

          // Get layers for this model
          const layersToCapture = FEATURE_EXPLORER_LAYERS[model] || [0];

          for (const layerNum of layersToCapture) {
            const isAllLayers = layerNum === 'all';
            console.log(`   🔧 Switching to ${isAllLayers ? 'all layers' : `layer ${layerNum}`}...`);

            // Change layer via sidebar dropdown
            const layerChanged = await page.evaluate((targetLayer, modelType, isAll) => {
              // Find the layer selector (usually in ConceptSelector or sidebar)
              const selects = document.querySelectorAll('select');
              for (const select of selects) {
                const options = Array.from(select.options);
                let layerOption;

                if (isAll) {
                  // Look for "All" option
                  layerOption = options.find(opt =>
                    opt.value === 'all_layers' ||
                    opt.value.includes('all_layers') ||
                    opt.text.toLowerCase() === 'all'
                  );
                } else {
                  // Look for layer options (e.g., "Layer 0", "L0", or just number)
                  layerOption = options.find(opt =>
                    opt.value.includes(`layer_${targetLayer}`) ||
                    opt.value.includes(`layer-${targetLayer}`) ||
                    opt.text.toLowerCase().includes(`layer ${targetLayer}`) ||
                    opt.value === String(targetLayer)
                  );
                }

                if (layerOption) {
                  select.value = layerOption.value;
                  select.dispatchEvent(new Event('change', { bubbles: true }));
                  return true;
                }
              }
              return false;
            }, layerNum, model, isAllLayers);

            if (layerChanged) {
              await wait(5000); // Wait for layer data to load
            }

            // Cycle through color modes
            for (const colorMode of COLOR_MODES) {
              // Skip layer color mode if not all_layers view (layer color shows which layer each point is from)
              if (colorMode === 'layer' && !isAllLayers) continue;

              console.log(`   🎨 Setting color mode: ${colorMode}`);

              // Change color mode via the dropdown
              await page.evaluate((mode) => {
                const selects = document.querySelectorAll('select');
                for (const select of selects) {
                  const options = Array.from(select.options);
                  const modeOption = options.find(opt =>
                    opt.value === mode ||
                    opt.text.toLowerCase().includes(mode)
                  );
                  if (modeOption) {
                    select.value = modeOption.value;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                    return true;
                  }
                }
                return false;
              }, colorMode);

              await wait(2000);

              // Zoom in by simulating scroll on the SVG
              console.log('   🔍 Zooming in...');
              const svgElement = await page.$('svg');
              if (svgElement) {
                const box = await svgElement.boundingBox();
                if (box) {
                  const centerX = box.x + box.width / 2;
                  const centerY = box.y + box.height / 2;

                  // Move mouse to center and zoom in with scroll
                  await page.mouse.move(centerX, centerY);
                  for (let i = 0; i < 4; i++) {
                    await page.mouse.wheel({ deltaY: -150 }); // Negative = zoom in
                    await wait(300);
                  }
                }
              }

              await wait(2000);

              // Click on a point to show feature details
              await page.evaluate(() => {
                const points = document.querySelectorAll('circle:not([r="0"]), .scatter-point');
                if (points.length > 0) {
                  // Click a point near the center
                  const centerIndex = Math.floor(points.length / 2);
                  points[centerIndex]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
                }
              });
              await wait(2000);

              // Take screenshot for this combination
              const layerLabel = isAllLayers ? 'all' : `L${layerNum}`;
              const featureFilename = model === 'pi05'
                ? `action-atlas_features_${layerLabel}_${colorMode}_6x.png`
                : `action-atlas_features_${model}_${layerLabel}_${colorMode}_6x.png`;
              const featureFilepath = path.join(CONFIG.outputDir, featureFilename);

              await page.screenshot({ path: featureFilepath, type: 'png', fullPage: false });

              const stats = fs.statSync(featureFilepath);
              console.log(`   ✅ Saved: ${featureFilename} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);

              // Reset zoom for next iteration
              await page.evaluate(() => {
                // Try to find and click a reset zoom button, or press Home key
                document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Home', bubbles: true }));
              });
              await wait(1000);
            }
          }

          // Continue to take the standard screenshot as well
          console.log('   📸 Taking standard feature view...');
        } else if (tab.id === 'features' && tab.needsClick) {
          // Original simple click behavior for backward compatibility
          await page.evaluate(() => {
            const points = document.querySelectorAll('circle, .scatter-point');
            if (points.length > 0) {
              points[Math.floor(points.length / 2)]?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
            }
          });
          await wait(3000);
        }

        // Scroll to show full content
        await page.evaluate(() => {
          const main = document.querySelector('main') || document.body;
          main.scrollTop = 0;
        });
        await wait(1000);

        // Screenshot with model suffix
        const filename = model === 'pi05'
          ? `action-atlas_${tab.id}_6x.png`
          : `action-atlas_${tab.id}_${model}_6x.png`;
        const filepath = path.join(CONFIG.outputDir, filename);

        await page.screenshot({ path: filepath, type: 'png', fullPage: false });

        const stats = fs.statSync(filepath);
        console.log(`   ✅ Saved: ${filename} (${(stats.size / 1024 / 1024).toFixed(2)} MB)`);
      }
    }

    console.log('\n✨ Complete!');
    console.log(`📁 ${CONFIG.outputDir}`);

  } catch (error) {
    console.error('\n❌ Error:', error.message);
    await page.screenshot({ path: path.join(CONFIG.outputDir, 'error.png') });
  } finally {
    await browser.close();
  }
}

captureScreenshots();
