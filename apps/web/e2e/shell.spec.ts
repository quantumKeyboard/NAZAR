import { expect, test } from '@playwright/test';
import { collectConsoleErrors } from './helpers';

test('main screen renders with live API status and no console errors', async ({ page }) => {
  const errors = collectConsoleErrors(page);
  await page.goto('/');

  await expect(page.getByRole('heading', { level: 1 })).toHaveText(/one pass/i);
  await expect(page.locator('#missions')).toBeVisible();

  // Real API: secrets are forced blank for E2E, so the API is up but honestly not ready.
  const status = page.getByTestId('api-status');
  await expect(status).toHaveAttribute('data-state', 'not-ready');
  await expect(status).toContainText('API online · not ready');

  const panel = page.getByTestId('readiness-panel');
  await expect(panel).toContainText('database');
  await expect(panel).toContainText('config:NAZAR_SECRET_KEY');
  await expect(panel).toContainText('Not configured · required');
  // The operator account was created by the setup project, so this check now passes.
  await expect(panel.getByRole('listitem').filter({ hasText: 'accounts' })).toContainText('OK');

  await expect(page.getByTestId('viewer-canvas').locator('canvas')).toBeVisible();
  await expect(page.getByTestId('viewer-empty')).toHaveText('No geometry yet.');

  await page.screenshot({ path: test.info().outputPath('main.png'), fullPage: true });
  expect(errors).toEqual([]);
});

test('API indicator reports offline when the health endpoint is unreachable', async ({ page }) => {
  await page.route('**/api/health', (route) => route.abort('connectionrefused'));
  await page.goto('/');
  const status = page.getByTestId('api-status');
  await expect(status).toHaveAttribute('data-state', 'offline');
  await expect(status).toContainText('API offline');
  await expect(page.getByTestId('readiness-panel')).toContainText('did not respond');
});

test('viewer controls toggle layers and expose pressed state', async ({ page }) => {
  await page.goto('/');
  const grid = page.getByRole('button', { name: 'Grid' });
  // The reference grid is off by default: it only distracted from the model.
  await expect(grid).toHaveAttribute('aria-pressed', 'false');
  await grid.click();
  await expect(grid).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Reset view' }).click();
  await expect(page.getByTestId('viewer-canvas').locator('canvas')).toBeVisible();
});

test('keyboard users can skip to content and see focus', async ({ page, isMobile }) => {
  test.skip(isMobile, 'Keyboard navigation is checked on desktop.');
  await page.goto('/');
  await page.keyboard.press('Tab');
  const skip = page.getByRole('link', { name: 'Skip to content' });
  await expect(skip).toBeFocused();
  await expect(skip).toBeVisible();
});

test('layout has no horizontal overflow', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('api-status')).not.toHaveAttribute('data-state', 'checking');
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
