"""Product registration and management routes."""

from __future__ import annotations

import logging
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from defectvision.db.database import get_db
from defectvision.db.models import Product, ProductStatus
from defectvision.schemas.product import ProductCreate, ProductResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/products", tags=["Products"])


@router.post(
    "",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new product",
)
def register_product(
    payload: ProductCreate,
    db: Session = Depends(get_db),
) -> Product:
    """Register an industrial product in DefectVision.

    Newly created products start in REGISTERED status and transition to READY
    after successful threshold calibration.
    """
    existing = db.execute(
        select(Product).where(Product.product_id == payload.product_id)
    ).scalar_one_or_none()

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Product with product_id '{payload.product_id}' already exists.",
        )

    product = Product(
        product_id=payload.product_id,
        name=payload.name,
        description=payload.description,
        status=ProductStatus.REGISTERED.value,
        metadata_json=payload.metadata,
    )
    db.add(product)
    db.commit()
    db.refresh(product)

    logger.info(f"Registered product '{product.product_id}' (id={product.id})")
    return product


@router.get(
    "",
    response_model=list[ProductResponse],
    summary="List all registered products",
)
def list_products(
    db: Session = Depends(get_db),
) -> Sequence[Product]:
    """Retrieve all registered products and their current calibration status."""
    stmt = select(Product).order_by(Product.created_at.desc())
    products = db.execute(stmt).scalars().all()
    return products


@router.get(
    "/{product_id}",
    response_model=ProductResponse,
    summary="Get product details by product ID",
)
def get_product(
    product_id: str,
    db: Session = Depends(get_db),
) -> Product:
    """Retrieve a single product by its unique product_id."""
    product = db.execute(
        select(Product).where(Product.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product '{product_id}' not found.",
        )

    return product
