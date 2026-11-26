"""登录接口。

设计取舍（这是一个"先能测、后面还要长"的学校场景）：

* **学号即账号**：``student_id`` 是 3–32 位字符串，校方可直接用学号。
* **首次登录自动建号**（``AUTH_AUTO_REGISTER=true``）：演示环境零配置；
  正式上线应改为校方导入名单并把开关关掉。
* **令牌存服务端**：能立即吊销，不依赖 JWT 的无状态性。
* 账号表留了 ``extra`` 扩展位，后续加年级/班级/班主任等字段不必重建表。
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth.security import mask_student_id
from app.auth.store import Account

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["auth"])

BEARER_PREFIX = "bearer "


# ----------------------------------------------------------------------
# 请求 / 响应模型
# ----------------------------------------------------------------------
class LoginRequest(BaseModel):
    """登录请求。"""

    student_id: str = Field(min_length=3, max_length=32, description="学号")
    password: str = Field(min_length=6, max_length=128, description="密码（≥6 位）")
    name: str = Field(default="", max_length=32, description="显示名，首次登录时可填")


class LoginResponse(BaseModel):
    """登录结果。"""

    token: str
    student_id: str
    name: str
    expires_at: float
    created: bool = Field(description="本次是否新建了账号")


class MeResponse(BaseModel):
    """当前登录者。"""

    student_id: str
    name: str


# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def _container(request: Request):
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return container


def extract_token(request: Request) -> str:
    """从 ``Authorization: Bearer xxx`` 取令牌，没有则返回空串。"""
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith(BEARER_PREFIX):
        return ""
    return header[len(BEARER_PREFIX) :].strip()


async def current_account(request: Request) -> Account:
    """取当前登录账号；未登录或令牌失效抛 401。"""
    token = extract_token(request)
    if not token:
        raise HTTPException(status_code=401, detail="未登录")
    account = await _container(request).auth.resolve(token)
    if account is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return account


async def resolve_student_id(request: Request, claimed: str) -> str:
    """确定这一轮对话属于哪个学生。

    带令牌时以令牌为准（前端声称的 ``student_id`` 不作数，防止冒用别人身份）；
    没带令牌时沿用请求体里的值 —— 保留给脚本/联调与既有测试。
    """
    token = extract_token(request)
    if not token:
        return claimed
    account = await _container(request).auth.resolve(token)
    if account is None:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录")
    return account.student_id


# ----------------------------------------------------------------------
# 路由
# ----------------------------------------------------------------------
@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, request: Request) -> LoginResponse:
    """登录；账号不存在时按配置自动创建。"""
    container = _container(request)
    store = container.auth
    settings = container.settings

    student_id = payload.student_id.strip()
    if not student_id:
        raise HTTPException(status_code=422, detail="学号不能为空")

    account = await store.get(student_id)
    created = False
    if account is None:
        if not settings.auth_auto_register:
            # 不区分"账号不存在"和"密码错误"，避免被用来枚举学号
            raise HTTPException(status_code=401, detail="学号或密码不正确")
        account = await store.create(student_id, payload.name.strip(), payload.password)
        created = True
        logger.info("新建账号：%s", mask_student_id(student_id))
    elif not account.check(payload.password):
        logger.warning("登录失败（密码不匹配）：%s", mask_student_id(student_id))
        raise HTTPException(status_code=401, detail="学号或密码不正确")

    # 顺手清掉过期令牌，避免表无限增长（只有 SQL 实现有这个方法）
    purge = getattr(store, "purge_expired", None)
    if purge is not None:
        try:
            await purge()
        except Exception as exc:  # noqa: BLE001
            logger.warning("清理过期令牌失败：%s", exc)

    ttl_seconds = max(60, int(settings.auth_token_ttl_days) * 86_400)
    token, expires_at = await store.issue_token(student_id, ttl_seconds)
    logger.info(
        "登录成功：%s（%s）", mask_student_id(student_id), "新账号" if created else "已有账号"
    )
    return LoginResponse(
        token=token,
        student_id=account.student_id,
        name=account.name,
        expires_at=expires_at,
        created=created,
    )


@router.post("/logout")
async def logout(request: Request) -> dict:
    """退出登录（吊销当前令牌）。"""
    token = extract_token(request)
    if not token:
        return {"ok": True}
    revoked = await _container(request).auth.revoke(token)
    return {"ok": True, "revoked": bool(revoked)}


@router.get("/me", response_model=MeResponse)
async def me(request: Request) -> MeResponse:
    """返回当前登录者。"""
    account = await current_account(request)
    return MeResponse(student_id=account.student_id, name=account.name)


__all__ = [
    "current_account",
    "extract_token",
    "resolve_student_id",
    "router",
]
