###########################################################################
##                            IMPORTS
###########################################################################

from typing import Optional
from sqlmodel import SQLModel, Field
from sqlalchemy import CheckConstraint


###########################################################################
##                  SQLMODEL TABLES -- SALES DB
###########################################################################

class ProductDB(SQLModel, table=True):
    __tablename__ = "products"
    __table_args__ = (CheckConstraint("production_cost >= 0", name="ck_products_cost"),)
    product_id: int = Field(primary_key=True)
    category: str
    sub_category: str
    description_pt: str
    description_de: str
    description_fr: str
    description_es: str
    description_en: str
    description_zh: str
    color: Optional[str] = None
    sizes: Optional[str] = None
    production_cost: float


class StoreDB(SQLModel, table=True):
    __tablename__ = "stores"
    store_id: int = Field(primary_key=True)
    country: str
    city: str
    store_name: str
    number_of_employees: int
    zip_code: str
    latitude: float
    longitude: float


class CustomerDB(SQLModel, table=True):
    __tablename__ = "customers"
    customer_id: int = Field(primary_key=True)
    name: str
    email: Optional[str] = None
    telephone: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None
    job_title: Optional[str] = None


class EmployeeDB(SQLModel, table=True):
    __tablename__ = "employees"
    employee_id: int = Field(primary_key=True)
    store_id: int = Field(foreign_key="stores.store_id")
    name: str
    position: str


class DiscountDB(SQLModel, table=True):
    __tablename__ = "discounts"
    id: Optional[int] = Field(default=None, primary_key=True)
    start: str
    end: str
    discount: float
    description: str
    category: Optional[str] = None
    sub_category: Optional[str] = None


class TransactionDB(SQLModel, table=True):
    __tablename__ = "transactions"
    __table_args__ = (CheckConstraint("quantity > 0", name="ck_transactions_quantity"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    invoice_id: str
    line: int
    customer_id: int = Field(foreign_key="customers.customer_id")
    product_id: int = Field(foreign_key="products.product_id")
    size: Optional[str] = None
    color: Optional[str] = None
    unit_price: float
    quantity: int
    date: str
    discount: float
    line_total: float
    store_id: int = Field(foreign_key="stores.store_id")
    employee_id: int = Field(foreign_key="employees.employee_id")
    currency: str
    currency_symbol: str
    sku: str
    transaction_type: str
    payment_method: str
    invoice_total: float



