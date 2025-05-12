import json
from contextlib import contextmanager
from typing import Generator, Optional, Type

from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .logging import logger

# Import other ORM models as needed for upsert logic
from .models import Base
from .models import Company as CompanyPydantic
from .models import (
    CompanyOrm,
    CompetitorOrm,
    ComplianceNewsOrm,
    CultureSignalOrm,
    ExecutiveOrm,
    ExtractionMetadataOrm,
    FounderOrm,
    FundingRoundOrm,
    GoToMarketChannelOrm,
    GrowthMetricOrm,
    MarketSegmentOrm,
    OpportunityOrm,
    PatentOrm,
    PressReleaseOrm,
    PricingStrategyOrm,
    ProductOrm,
    RevenueModelOrm,
    RiskOrm,
    SourceOrm,
    SWOTEntryOrm,
    TagOrm,
    TechnologyOrm,
)

# Define the engine globally or within a function based on application structure
# Ensure the DATABASE_URL is correctly configured in settings
# Example: engine = create_engine(settings.DATABASE_URL, echo=settings.DB_ECHO)
# Using defaults if not in settings for now:
DB_URL = getattr(settings, 'DATABASE_URL', 'sqlite:///./startups.db')
DB_ECHO = getattr(settings, 'DB_ECHO', False)
engine = create_engine(DB_URL, echo=DB_ECHO)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database transaction failed: {e}")
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initializes the database by creating tables based on SQLAlchemy models."""
    logger.info(f"Initializing database: {DB_URL}")  # Use derived DB_URL
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(
            "Database tables created successfully (if they didn't exist).")
    except SQLAlchemyError as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def pydantic_to_orm(pydantic_obj,
                    orm_class: Type[Base],
                    existing_orm_obj=None,
                    **extra_fields):
    """Helper to convert Pydantic model to SQLAlchemy ORM instance."""
    data = pydantic_obj.model_dump(exclude_unset=True, exclude_none=True)
    data.update(extra_fields)  # Add foreign keys etc.

    # Remove fields that are relationships or not direct columns in this ORM class
    fields_to_remove = []
    if hasattr(orm_class, '__mapper__'):
        orm_columns = {c.key for c in orm_class.__mapper__.columns}
        relationship_keys = set(orm_class.__mapper__.relationships.keys())
        fields_to_remove = [
            k for k in data if k not in orm_columns
            and k not in relationship_keys and k != 'embedding_json'
        ]  # keep embedding_json if Pydantic has it

    # Also remove Pydantic fields corresponding to relationships handled separately
    if hasattr(pydantic_obj, '__fields__'):
        for field_name, field in pydantic_obj.__fields__.items():
            if isinstance(field.outer_type_, type) and issubclass(
                    field.outer_type_, list):
                # Basic check for list fields which often represent relationships
                # More robust check might involve inspecting the list item type
                if field_name in data:
                    fields_to_remove.append(field_name)

    for key in set(fields_to_remove):  # Use set to avoid duplicates
        if key in data:
            del data[key]

    if existing_orm_obj:
        for key, value in data.items():
            setattr(existing_orm_obj, key, value)
        return existing_orm_obj
    else:
        return orm_class(**data)


def upsert_company_data(company_data: CompanyPydantic) -> Optional[CompanyOrm]:
    """
    Upserts company data into the database based on the Pydantic model.
    Handles nested relationships. Finds company by name for upsert.
    """
    logger.info(f"Attempting to upsert data for company: {company_data.name}")
    with get_db_session() as db:
        try:
            # Check if company exists
            stmt = select(CompanyOrm).where(
                CompanyOrm.name == company_data.name)
            existing_company = db.execute(stmt).scalar_one_or_none()

            if existing_company:
                logger.info(f"Updating existing company: {company_data.name}")
                # Update top-level company fields
                company_orm = pydantic_to_orm(
                    company_data,
                    CompanyOrm,
                    existing_orm_obj=existing_company)
                # Clear existing relationships to replace them (cascade delete should handle removal)
                # This is a simple replacement strategy. More sophisticated merging might be needed.
                existing_company.founders.clear()
                existing_company.funding_rounds.clear()
                existing_company.products.clear()
                existing_company.patents.clear()
                # ... clear other relationships ...
                existing_company.tags.clear()
                existing_company.extraction_metadata.clear()
                existing_company.sources.clear()

            else:
                logger.info(f"Inserting new company: {company_data.name}")
                company_orm = pydantic_to_orm(company_data, CompanyOrm)
                db.add(company_orm)
                # Flush to get the company_id for relationships if it's new
                db.flush()

            # --- Handle Relationships ---
            # Note: This assumes full replacement of related items on update.

            # Simple One-to-Many (like Founders, FundingRounds, etc.)
            for founder_pydantic in company_data.founders:
                founder_orm = pydantic_to_orm(
                    founder_pydantic,
                    FounderOrm,
                    company_id=company_orm.company_id)
                company_orm.founders.append(founder_orm)

            for fr_pydantic in company_data.funding_rounds:
                fr_orm = pydantic_to_orm(fr_pydantic,
                                         FundingRoundOrm,
                                         company_id=company_orm.company_id)
                company_orm.funding_rounds.append(fr_orm)

            for patent_pydantic in company_data.patents:
                patent_orm = pydantic_to_orm(patent_pydantic,
                                             PatentOrm,
                                             company_id=company_orm.company_id)
                company_orm.patents.append(patent_orm)

            for segment_pydantic in company_data.market_segments:
                segment_orm = pydantic_to_orm(
                    segment_pydantic,
                    MarketSegmentOrm,
                    company_id=company_orm.company_id)
                company_orm.market_segments.append(segment_orm)

            for rm_pydantic in company_data.revenue_models:
                rm_orm = pydantic_to_orm(rm_pydantic,
                                         RevenueModelOrm,
                                         company_id=company_orm.company_id)
                company_orm.revenue_models.append(rm_orm)

            for ps_pydantic in company_data.pricing_strategies:
                ps_orm = pydantic_to_orm(ps_pydantic,
                                         PricingStrategyOrm,
                                         company_id=company_orm.company_id)
                company_orm.pricing_strategies.append(ps_orm)

            for gtm_pydantic in company_data.go_to_market_channels:
                gtm_orm = pydantic_to_orm(gtm_pydantic,
                                          GoToMarketChannelOrm,
                                          company_id=company_orm.company_id)
                company_orm.go_to_market_channels.append(gtm_orm)

            for comp_pydantic in company_data.competitors:
                comp_orm = pydantic_to_orm(comp_pydantic,
                                           CompetitorOrm,
                                           company_id=company_orm.company_id)
                company_orm.competitors.append(comp_orm)

            for swot_pydantic in company_data.swot_entries:
                swot_orm = pydantic_to_orm(swot_pydantic,
                                           SWOTEntryOrm,
                                           company_id=company_orm.company_id)
                company_orm.swot_entries.append(swot_orm)

            for pr_pydantic in company_data.press_releases:
                pr_orm = pydantic_to_orm(pr_pydantic,
                                         PressReleaseOrm,
                                         company_id=company_orm.company_id)
                company_orm.press_releases.append(pr_orm)

            for gm_pydantic in company_data.growth_metrics:
                gm_orm = pydantic_to_orm(gm_pydantic,
                                         GrowthMetricOrm,
                                         company_id=company_orm.company_id)
                company_orm.growth_metrics.append(gm_orm)

            for cn_pydantic in company_data.compliance_news:
                cn_orm = pydantic_to_orm(cn_pydantic,
                                         ComplianceNewsOrm,
                                         company_id=company_orm.company_id)
                company_orm.compliance_news.append(cn_orm)

            for exec_pydantic in company_data.executives:
                exec_orm = pydantic_to_orm(exec_pydantic,
                                           ExecutiveOrm,
                                           company_id=company_orm.company_id)
                company_orm.executives.append(exec_orm)

            for cs_pydantic in company_data.culture_signals:
                cs_orm = pydantic_to_orm(cs_pydantic,
                                         CultureSignalOrm,
                                         company_id=company_orm.company_id)
                company_orm.culture_signals.append(cs_orm)

            for risk_pydantic in company_data.risks:
                risk_orm = pydantic_to_orm(risk_pydantic,
                                           RiskOrm,
                                           company_id=company_orm.company_id)
                company_orm.risks.append(risk_orm)

            for opp_pydantic in company_data.opportunities:
                opp_orm = pydantic_to_orm(opp_pydantic,
                                          OpportunityOrm,
                                          company_id=company_orm.company_id)
                company_orm.opportunities.append(opp_orm)

            for tag_pydantic in company_data.tags:
                # Assuming TagOrm relates directly to CompanyOrm for now
                tag_orm = pydantic_to_orm(tag_pydantic,
                                          TagOrm,
                                          company_id=company_orm.company_id)
                company_orm.tags.append(tag_orm)

            for meta_pydantic in company_data.extraction_metadata:
                meta_orm = pydantic_to_orm(meta_pydantic,
                                           ExtractionMetadataOrm,
                                           company_id=company_orm.company_id)
                company_orm.extraction_metadata.append(meta_orm)

            for source_pydantic in company_data.sources:
                source_orm = pydantic_to_orm(source_pydantic,
                                             SourceOrm,
                                             company_id=company_orm.company_id)
                company_orm.sources.append(source_orm)

            # Products and Technologies (Many-to-Many via ProductOrm)
            for product_pydantic in company_data.products:
                # Check/create technologies first (assuming they are independent or looked up)
                tech_orms = []
                for tech_pydantic in product_pydantic.technologies:
                    stmt_tech = select(TechnologyOrm).where(
                        TechnologyOrm.name == tech_pydantic.name)
                    existing_tech = db.execute(stmt_tech).scalar_one_or_none()
                    if existing_tech:
                        tech_orms.append(existing_tech)
                    else:
                        new_tech_orm = pydantic_to_orm(tech_pydantic,
                                                       TechnologyOrm)
                        db.add(new_tech_orm)
                        # Flush to potentially get ID if needed elsewhere, though maybe not needed here
                        # db.flush()
                        tech_orms.append(new_tech_orm)

                # Create product ORM, linking technologies
                product_orm = pydantic_to_orm(
                    product_pydantic,
                    ProductOrm,
                    company_id=company_orm.company_id)
                product_orm.technologies = tech_orms  # Assign the list of TechnologyOrm objects
                company_orm.products.append(product_orm)

            # Commit happens automatically via context manager `get_db_session` exit
            db.flush()  # Ensure IDs are populated before returning
            db.refresh(company_orm
                       )  # Refresh to get latest state including relationships
            logger.info(
                f"Successfully upserted company: {company_orm.name} (ID: {company_orm.company_id})"
            )
            return company_orm

        except SQLAlchemyError as e:
            logger.error(f"Database error during upsert for "
                         f"company {company_data.name}: {e}")
            # Rollback is handled by context manager
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error during upsert for "
                f"company {company_data.name}: {e}",
                exc_info=True)
            # Rollback is handled by context manager
            return None


# Example usage (optional, for testing)
if __name__ == '__main__':
    # Ensure you have a valid DATABASE_URL in your .env or config
    print("Initializing DB...")
    init_db()
    print("DB Initialized.")

    # # Example Pydantic data (replace with actual data)
    # test_company = CompanyPydantic(
    #     name="Test Startup Inc.",
    #     website="https://teststartup.com",
    #     founders=[Founder(name="Alice"), Founder(name="Bob", background="Engineer")],
    #     products=[Product(name="TestProduct", technologies=[Technology(name="Python"), Technology(name="AI")])]
    # )

    # print(f"Upserting {test_company.name}...")
    # inserted_company = upsert_company_data(test_company)
    # if inserted_company:
    #      print(f"Upsert successful. Company ID: {inserted_company.company_id}")
    #      print(f"  Founders: {len(inserted_company.founders)}")
    #      print(f"  Products: {len(inserted_company.products)}")
    #      if inserted_company.products:
    #          print(f"    Product 1 Technologies: {[t.name for t in inserted_company.products[0].technologies]}")

    # else:
    #     print("Upsert failed.")
    # # Example Pydantic data (replace with actual data)
    # test_company = CompanyPydantic(
    #     name="Test Startup Inc.",
    #     website="https://teststartup.com",
    #     founders=[Founder(name="Alice"), Founder(name="Bob", background="Engineer")],
    #     products=[Product(name="TestProduct", technologies=[Technology(name="Python"), Technology(name="AI")])]
    # )

    # print(f"Upserting {test_company.name}...")
    # inserted_company = upsert_company_data(test_company)
    # if inserted_company:
    #      print(f"Upsert successful. Company ID: {inserted_company.company_id}")
    #      print(f"  Founders: {len(inserted_company.founders)}")
    #      print(f"  Products: {len(inserted_company.products)}")
    #      if inserted_company.products:
    #          print(f"    Product 1 Technologies: {[t.name for t in inserted_company.products[0].technologies]}")

    # else:
    #     print("Upsert failed.")
