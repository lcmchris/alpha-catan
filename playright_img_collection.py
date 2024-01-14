from concurrent.futures.thread import ThreadPoolExecutor
import asyncio
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright, Playwright
import time

import threading
from playwright.sync_api import sync_playwright
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class Tls(threading.local):
    def __init__(self) -> None:
        self.playwright = sync_playwright().start()
        print("Create playwright instance in Thread", threading.current_thread().name)


class Worker:
    tls = Tls()

    def run(self, idx):
        try:
            print("Launched worker in ", threading.current_thread().name)
            browser = self.tls.playwright.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()

            page.goto("https://colonist.io/")
            page.click("text=Agree")
            page.click("text=Play vs Bots")
            canvas = page.locator("canvas").nth(0)
            canvas.click()
            time.sleep(1)

            canvas.screenshot(path=f"data/{idx}_playwright_img_collection.png")
            time.sleep(5)
            context.close()
            browser.close()
            print("Stopped worker in ", threading.current_thread().name)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx in range(240, 1000):
            worker = Worker()
            executor.submit(worker.run(idx))


# def sync_run(idx: int):
#     with sync_playwright() as playwright:
#         chromium = playwright.chromium


# async def run(playwright: Playwright, idx: int):
#     chromium = playwright.chromium  # or "firefox" or "webkit".
#     browser = await chromium.launch(headless=False)
#     page = await browser.new_page()
#     await page.goto("https://colonist.io/")
#     await page.click("text=Agree")
#     await page.click("text=Play vs Bots")
#     await page.locator("canvas").nth(0).screenshot(
#         path=f"{idx}_playwright_img_collection.png"
#     )
#     time.sleep(3)
#     await browser.close()


# async def main(idx: int):
#     async with async_playwright() as playwright:
#         await run(playwright, idx)


# with ThreadPoolExecutor(max_workers=10) as executor:
#     for idx in range(1000):
#         executor.submit(sync_run(idx))
