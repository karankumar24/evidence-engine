"""Design system gallery — dev-only, gated by settings.debug."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from evidenceengine.core.config import settings

router = APIRouter(tags=["design-system"])

_templates = Jinja2Templates(directory=str(Path(__file__).parent.parent.parent / "templates"))


@router.get("/design-system", response_class=HTMLResponse)
async def design_system(request: Request) -> HTMLResponse:
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    return _templates.TemplateResponse(request, "design_system.html")
