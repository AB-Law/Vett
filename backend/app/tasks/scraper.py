"""Phase 2 – Celery scraper tasks (scaffolded).

Requires:
  - Playwright installed: playwright install chromium
  - Celery worker running
  - Redis broker

Usage (when enabled):
  from app.tasks.scraper import scrape_jobs_task
  task_id = scrape_jobs_task.delay(query="ML Engineer", location="Remote", source="linkedin")
"""
# from app.celery_app import celery_app
# from playwright.async_api import async_playwright
# import asyncio


# @celery_app.task(bind=True)
# def scrape_jobs_task(self, query: str, location: str = None, source: str = "all"):
#     """Scrape jobs and score each one against the current CV."""
#     return asyncio.run(_scrape_and_score(self, query, location, source))


# async def _scrape_and_score(task, query, location, source):
#     jobs = []
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         if source in ("linkedin", "all"):
#             jobs += await _scrape_linkedin(browser, query, location)
#         if source in ("indeed", "all"):
#             jobs += await _scrape_indeed(browser, query, location)
#         if source in ("naukri", "all"):
#             jobs += await _scrape_naukri(browser, query, location)
#         await browser.close()
#     # Score each job against CV and persist...
#     return {"scraped": len(jobs)}


# Phase 2 placeholder
def scrape_jobs_placeholder():
    raise NotImplementedError("Phase 2: enable Celery worker and implement scrapers")
