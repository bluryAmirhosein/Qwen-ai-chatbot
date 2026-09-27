from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, LargeBinary, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Raw float32 bytes, same layout as the old SQLite BLOB
    # (np.float32.tobytes()) — kept identical so embed/search code doesn't change.
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")