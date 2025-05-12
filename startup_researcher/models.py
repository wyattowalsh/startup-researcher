import json
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    create_engine,
)
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship
from sqlalchemy.sql import func

# --- Pydantic Models (from docs.md) ---


class Founder(BaseModel):
    founder_id: Optional[int] = None  # Made optional for creation
    name: str
    background: Optional[str] = None


class FundingRound(BaseModel):
    funding_round_id: Optional[int] = None  # Made optional for creation
    round: str
    amount: Optional[float] = None
    currency: str = 'USD'
    date: Optional[date] = None
    lead_investor: Optional[str] = None


class Technology(BaseModel):
    technology_id: Optional[int] = None  # Made optional for creation
    name: str
    description: Optional[str] = None


class Patent(BaseModel):
    patent_id: Optional[int] = None  # Made optional for creation
    patent_number: str
    title: Optional[str] = None
    filing_date: Optional[date] = None
    status: Optional[str] = None


class Product(BaseModel):
    product_id: Optional[int] = None  # Made optional for creation
    name: str
    description: Optional[str] = None
    value_proposition: Optional[str] = None
    version_number: int = 1
    is_latest: bool = True
    sentiment_score: Optional[float] = None
    # embedding: Optional[List[float]] = None # Embeddings might be stored separately or as JSON
    embedding_json: Optional[str] = None  # Store as JSON string
    auto_category: Optional[str] = None
    importance_score: Optional[float] = None
    keywords: List[str] = []
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]
    technologies: List[Technology] = []

    def set_embedding(self, embedding: List[float]):
        self.embedding_json = json.dumps(embedding)

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None


class MarketSegment(BaseModel):
    segment_id: Optional[int] = None  # Made optional for creation
    name: str
    description: Optional[str] = None


class RevenueModel(BaseModel):
    revenue_model_id: Optional[int] = None  # Made optional for creation
    type: str
    description: Optional[str] = None


class PricingStrategy(BaseModel):
    pricing_strategy_id: Optional[int] = None  # Made optional for creation
    strategy: str
    details: Optional[str] = None


class GoToMarketChannel(BaseModel):
    channel_id: Optional[int] = None  # Made optional for creation
    channel: str
    description: Optional[str] = None


class Competitor(BaseModel):
    competitor_id: Optional[int] = None  # Made optional for creation
    name: str
    description: Optional[str] = None


class SWOTEntry(BaseModel):
    swot_id: Optional[int] = None  # Made optional for creation
    category: Literal['Strength', 'Weakness', 'Opportunity', 'Threat']
    description: str


class PressRelease(BaseModel):
    press_release_id: Optional[int] = None  # Made optional for creation
    version_number: int = 1
    is_latest: bool = True
    title: str
    url: Optional[str] = None
    publication_date: Optional[date] = None
    summary: Optional[str] = None
    sentiment_score: Optional[float] = None
    # embedding: Optional[List[float]] = None
    embedding_json: Optional[str] = None  # Store as JSON string
    auto_category: Optional[str] = None
    importance_score: Optional[float] = None
    keywords: List[str] = []
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]

    def set_embedding(self, embedding: List[float]):
        self.embedding_json = json.dumps(embedding)

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None


class GrowthMetric(BaseModel):
    metric_id: Optional[int] = None  # Made optional for creation
    metric_name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    as_of_date: Optional[date] = None
    version_number: int = 1
    is_latest: bool = True


class ComplianceNews(BaseModel):
    compliance_id: Optional[int] = None  # Made optional for creation
    title: str
    url: Optional[str] = None
    publication_date: Optional[date] = None
    description: Optional[str] = None
    sentiment_score: Optional[float] = None
    # embedding: Optional[List[float]] = None
    embedding_json: Optional[str] = None  # Store as JSON string
    auto_category: Optional[str] = None
    importance_score: Optional[float] = None
    keywords: List[str] = []
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]

    def set_embedding(self, embedding: List[float]):
        self.embedding_json = json.dumps(embedding)

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None


class Executive(BaseModel):
    executive_id: Optional[int] = None  # Made optional for creation
    name: str
    title: Optional[str] = None
    achievements: Optional[str] = None


class CultureSignal(BaseModel):
    signal_id: Optional[int] = None  # Made optional for creation
    source: str
    signal_type: Optional[str] = None
    content: str
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    # embedding: Optional[List[float]] = None
    embedding_json: Optional[str] = None  # Store as JSON string
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]

    def set_embedding(self, embedding: List[float]):
        self.embedding_json = json.dumps(embedding)

    def get_embedding(self) -> Optional[List[float]]:
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None


class Risk(BaseModel):
    risk_id: Optional[int] = None  # Made optional for creation
    description: str
    category: Optional[str] = None
    likelihood: Optional[str] = None
    impact: Optional[str] = None
    version_number: int = 1
    is_latest: bool = True
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]


class Opportunity(BaseModel):
    opportunity_id: Optional[int] = None  # Made optional for creation
    description: str
    recommended_action: Optional[str] = None
    raw_extraction: Optional[Dict[
        str, Any]] = None  # Changed from dict to Dict[str, Any]


class Tag(BaseModel):
    tag_id: Optional[int] = None  # Made optional for creation
    entity_type: str
    entity_id: int  # Assuming this refers to the DB ID of the related entity
    tag: str


class ExtractionMetadata(BaseModel):
    metadata_id: Optional[int] = None  # Made optional for creation
    entity_type: str
    entity_id: int  # Assuming this refers to the DB ID of the related entity
    extraction_date: datetime
    confidence_score: Optional[float] = None
    llm_prompt: Optional[str] = None
    llm_version: Optional[str] = None


class Source(BaseModel):
    source_id: Optional[int] = None  # Made optional for creation
    entity_type: str
    entity_id: int  # Assuming this refers to the DB ID of the related entity
    source_name: str
    source_url: Optional[str] = None
    publication_date: Optional[date] = None


class Company(BaseModel):
    company_id: Optional[int] = None  # Made optional for creation
    name: str  # Unique company name
    website: Optional[str] = None
    founding_date: Optional[date] = None
    headquarters: Optional[str] = None
    legal_entity: Optional[str] = None
    founders: List[Founder] = []
    funding_rounds: List[FundingRound] = []
    products: List[Product] = []
    patents: List[Patent] = []
    market_segments: List[MarketSegment] = []
    revenue_models: List[RevenueModel] = []
    pricing_strategies: List[PricingStrategy] = []
    go_to_market_channels: List[GoToMarketChannel] = []
    competitors: List[Competitor] = []
    swot_entries: List[SWOTEntry] = []
    press_releases: List[PressRelease] = []
    growth_metrics: List[GrowthMetric] = []
    compliance_news: List[ComplianceNews] = []
    executives: List[Executive] = []
    culture_signals: List[CultureSignal] = []
    risks: List[Risk] = []
    opportunities: List[Opportunity] = []
    tags: List[Tag] = []  # Note: Relationship managed separately in DB
    extraction_metadata: List[ExtractionMetadata] = [
    ]  # Note: Relationship managed separately in DB
    sources: List[Source] = []  # Note: Relationship managed separately in DB


# --- SQLAlchemy Base ---
Base = declarative_base()

# --- Association Tables (for Many-to-Many relationships if needed, e.g., Product <-> Technology) ---
product_technology_association = Table(
    'product_technology_association', Base.metadata,
    Column('product_id',
           Integer,
           ForeignKey('products.product_id'),
           primary_key=True),
    Column('technology_id',
           Integer,
           ForeignKey('technologies.technology_id'),
           primary_key=True))

# --- SQLAlchemy Models ---


class CompanyOrm(Base):
    __tablename__ = 'companies'
    company_id: Mapped[int] = mapped_column(Integer,
                                            primary_key=True,
                                            autoincrement=True)
    name: Mapped[str] = mapped_column(String,
                                      unique=True,
                                      index=True,
                                      nullable=False)
    website: Mapped[Optional[str]] = mapped_column(String)
    founding_date: Mapped[Optional[date]] = mapped_column(Date)
    headquarters: Mapped[Optional[str]] = mapped_column(String)
    legal_entity: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 onupdate=func.now(),
                                                 server_default=func.now())

    # Relationships (One-to-Many)
    founders: Mapped[List["FounderOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    funding_rounds: Mapped[List["FundingRoundOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    products: Mapped[List["ProductOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    patents: Mapped[List["PatentOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    market_segments: Mapped[List["MarketSegmentOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    revenue_models: Mapped[List["RevenueModelOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    pricing_strategies: Mapped[List["PricingStrategyOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    go_to_market_channels: Mapped[List["GoToMarketChannelOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    competitors: Mapped[List["CompetitorOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    swot_entries: Mapped[List["SWOTEntryOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    press_releases: Mapped[List["PressReleaseOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    growth_metrics: Mapped[List["GrowthMetricOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    compliance_news: Mapped[List["ComplianceNewsOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    executives: Mapped[List["ExecutiveOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    culture_signals: Mapped[List["CultureSignalOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    risks: Mapped[List["RiskOrm"]] = relationship(back_populates="company",
                                                  cascade="all, delete-orphan")
    opportunities: Mapped[List["OpportunityOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    tags: Mapped[List["TagOrm"]] = relationship(back_populates="company",
                                                cascade="all, delete-orphan")
    extraction_metadata: Mapped[List["ExtractionMetadataOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")
    sources: Mapped[List["SourceOrm"]] = relationship(
        back_populates="company", cascade="all, delete-orphan")


class FounderOrm(Base):
    __tablename__ = 'founders'
    founder_id: Mapped[int] = mapped_column(Integer,
                                            primary_key=True,
                                            autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    name: Mapped[str] = mapped_column(String, nullable=False)
    background: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(back_populates="founders")


class FundingRoundOrm(Base):
    __tablename__ = 'funding_rounds'
    funding_round_id: Mapped[int] = mapped_column(Integer,
                                                  primary_key=True,
                                                  autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    round: Mapped[str] = mapped_column(String)
    amount: Mapped[Optional[float]] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String, default='USD')
    date: Mapped[Optional[date]] = mapped_column(Date)
    lead_investor: Mapped[Optional[str]] = mapped_column(String)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="funding_rounds")


class TechnologyOrm(Base):
    __tablename__ = 'technologies'
    technology_id: Mapped[int] = mapped_column(Integer,
                                               primary_key=True,
                                               autoincrement=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    # Relationship for Many-to-Many with ProductOrm
    products: Mapped[List["ProductOrm"]] = relationship(
        secondary=product_technology_association,
        back_populates="technologies")


class PatentOrm(Base):
    __tablename__ = 'patents'
    patent_id: Mapped[int] = mapped_column(Integer,
                                           primary_key=True,
                                           autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    patent_number: Mapped[str] = mapped_column(String,
                                               nullable=False,
                                               unique=True)
    title: Mapped[Optional[str]] = mapped_column(String)
    filing_date: Mapped[Optional[date]] = mapped_column(Date)
    status: Mapped[Optional[str]] = mapped_column(String)
    company: Mapped["CompanyOrm"] = relationship(back_populates="patents")


class ProductOrm(Base):
    __tablename__ = 'products'
    product_id: Mapped[int] = mapped_column(Integer,
                                            primary_key=True,
                                            autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    value_proposition: Mapped[Optional[str]] = mapped_column(Text)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    embedding_json: Mapped[Optional[str]] = mapped_column(
        Text)  # Store embedding as JSON string
    auto_category: Mapped[Optional[str]] = mapped_column(String)
    importance_score: Mapped[Optional[float]] = mapped_column(Float)
    keywords: Mapped[Optional[str]] = mapped_column(
        Text)  # Store list as comma-separated or JSON
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(back_populates="products")
    # Relationship for Many-to-Many with TechnologyOrm
    technologies: Mapped[List["TechnologyOrm"]] = relationship(
        secondary=product_technology_association, back_populates="products")


class MarketSegmentOrm(Base):
    __tablename__ = 'market_segments'
    segment_id: Mapped[int] = mapped_column(Integer,
                                            primary_key=True,
                                            autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="market_segments")


class RevenueModelOrm(Base):
    __tablename__ = 'revenue_models'
    revenue_model_id: Mapped[int] = mapped_column(Integer,
                                                  primary_key=True,
                                                  autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    type: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="revenue_models")


class PricingStrategyOrm(Base):
    __tablename__ = 'pricing_strategies'
    pricing_strategy_id: Mapped[int] = mapped_column(Integer,
                                                     primary_key=True,
                                                     autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    strategy: Mapped[str] = mapped_column(String, nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="pricing_strategies")


class GoToMarketChannelOrm(Base):
    __tablename__ = 'go_to_market_channels'
    channel_id: Mapped[int] = mapped_column(Integer,
                                            primary_key=True,
                                            autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    channel: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="go_to_market_channels")


class CompetitorOrm(Base):
    __tablename__ = 'competitors'
    competitor_id: Mapped[int] = mapped_column(Integer,
                                               primary_key=True,
                                               autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(back_populates="competitors")


class SWOTEntryOrm(Base):
    __tablename__ = 'swot_entries'
    swot_id: Mapped[int] = mapped_column(Integer,
                                         primary_key=True,
                                         autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    category: Mapped[str] = mapped_column(
        String, nullable=False)  # Store Literal as string
    description: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped["CompanyOrm"] = relationship(back_populates="swot_entries")


class PressReleaseOrm(Base):
    __tablename__ = 'press_releases'
    press_release_id: Mapped[int] = mapped_column(Integer,
                                                  primary_key=True,
                                                  autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String)
    publication_date: Mapped[Optional[date]] = mapped_column(Date)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text)
    auto_category: Mapped[Optional[str]] = mapped_column(String)
    importance_score: Mapped[Optional[float]] = mapped_column(Float)
    keywords: Mapped[Optional[str]] = mapped_column(
        Text)  # Store list as comma-separated or JSON
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="press_releases")


class GrowthMetricOrm(Base):
    __tablename__ = 'growth_metrics'
    metric_id: Mapped[int] = mapped_column(Integer,
                                           primary_key=True,
                                           autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Float)
    unit: Mapped[Optional[str]] = mapped_column(String)
    as_of_date: Mapped[Optional[date]] = mapped_column(Date)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="growth_metrics")


class ComplianceNewsOrm(Base):
    __tablename__ = 'compliance_news'
    compliance_id: Mapped[int] = mapped_column(Integer,
                                               primary_key=True,
                                               autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    title: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String)
    publication_date: Mapped[Optional[date]] = mapped_column(Date)
    description: Mapped[Optional[str]] = mapped_column(Text)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text)
    auto_category: Mapped[Optional[str]] = mapped_column(String)
    importance_score: Mapped[Optional[float]] = mapped_column(Float)
    keywords: Mapped[Optional[str]] = mapped_column(
        Text)  # Store list as comma-separated or JSON
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="compliance_news")


class ExecutiveOrm(Base):
    __tablename__ = 'executives'
    executive_id: Mapped[int] = mapped_column(Integer,
                                              primary_key=True,
                                              autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    name: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String)
    achievements: Mapped[Optional[str]] = mapped_column(Text)
    company: Mapped["CompanyOrm"] = relationship(back_populates="executives")


class CultureSignalOrm(Base):
    __tablename__ = 'culture_signals'
    signal_id: Mapped[int] = mapped_column(Integer,
                                           primary_key=True,
                                           autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    source: Mapped[str] = mapped_column(String, nullable=False)
    signal_type: Mapped[Optional[str]] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[Optional[str]] = mapped_column(String)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float)
    embedding_json: Mapped[Optional[str]] = mapped_column(Text)
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="culture_signals")


class RiskOrm(Base):
    __tablename__ = 'risks'
    risk_id: Mapped[int] = mapped_column(Integer,
                                         primary_key=True,
                                         autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String)
    likelihood: Mapped[Optional[str]] = mapped_column(String)
    impact: Mapped[Optional[str]] = mapped_column(String)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=True)
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(back_populates="risks")


class OpportunityOrm(Base):
    __tablename__ = 'opportunities'
    opportunity_id: Mapped[int] = mapped_column(Integer,
                                                primary_key=True,
                                                autoincrement=True)
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.company_id'))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[Optional[str]] = mapped_column(Text)
    raw_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="opportunities")


class TagOrm(Base):
    __tablename__ = 'tags'
    tag_id: Mapped[int] = mapped_column(Integer,
                                        primary_key=True,
                                        autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey('companies.company_id'
                   ))  # Assuming tags only relate to Company for now
    # If tags can relate to other entities, need a more generic approach (e.g., polymorphic)
    # entity_type: Mapped[str] = mapped_column(String, nullable=False)
    # entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    tag: Mapped[str] = mapped_column(String, nullable=False)
    company: Mapped["CompanyOrm"] = relationship(back_populates="tags")


class ExtractionMetadataOrm(Base):
    __tablename__ = 'extraction_metadata'
    metadata_id: Mapped[int] = mapped_column(Integer,
                                             primary_key=True,
                                             autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey('companies.company_id'
                   ))  # Assuming metadata only relates to Company for now
    # entity_type: Mapped[str] = mapped_column(String, nullable=False)
    # entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    extraction_date: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                      nullable=False,
                                                      default=datetime.utcnow)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    llm_prompt: Mapped[Optional[str]] = mapped_column(Text)
    llm_version: Mapped[Optional[str]] = mapped_column(String)
    company: Mapped["CompanyOrm"] = relationship(
        back_populates="extraction_metadata")


class SourceOrm(Base):
    __tablename__ = 'sources'
    source_id: Mapped[int] = mapped_column(Integer,
                                           primary_key=True,
                                           autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey('companies.company_id'
                   ))  # Assuming sources only relate to Company for now
    # entity_type: Mapped[str] = mapped_column(String, nullable=False)
    # entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_name: Mapped[str] = mapped_column(String, nullable=False)
    source_url: Mapped[Optional[str]] = mapped_column(String)
    publication_date: Mapped[Optional[date]] = mapped_column(Date)
    company: Mapped["CompanyOrm"] = relationship(back_populates="sources")
