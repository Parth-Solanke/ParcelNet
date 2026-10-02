import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from backend.app.core.db import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    crs_epsg = Column(Integer, nullable=False, default=32643)
    bbox = Column(JSON, nullable=True)  # [minx, miny, maxx, maxy]
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(50), default="created", nullable=False)  # created, processing, ready, failed

    # Relationships
    rasters = relationship("Raster", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    parcels = relationship("Parcel", back_populates="project", cascade="all, delete-orphan")
    buildings = relationship("Building", back_populates="project", cascade="all, delete-orphan")
    roads = relationship("Road", back_populates="project", cascade="all, delete-orphan")
    landuses = relationship("LandUse", back_populates="project", cascade="all, delete-orphan")
    validation_issues = relationship("ValidationIssue", back_populates="project", cascade="all, delete-orphan")
    gt_points = relationship("GTPoint", back_populates="project", cascade="all, delete-orphan")
    control_points = relationship("ControlPoint", back_populates="project", cascade="all, delete-orphan")


class Raster(Base):
    __tablename__ = "rasters"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(20), nullable=False)  # ORI, DSM, DTM, NDSM, STACK
    path = Column(Text, nullable=False)
    gsd = Column(Float, nullable=False)
    bounds = Column(JSON, nullable=False)  # [minx, miny, maxx, maxy]
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="rasters")


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(100), nullable=False)  # ingest, preprocess, tile, infer, vectorize, topology, validate
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    progress = Column(Float, default=0.0, nullable=False)
    params_json = Column(JSON, default=dict)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    log = Column(Text, nullable=True)

    project = relationship("Project", back_populates="jobs")


class Parcel(Base):
    __tablename__ = "parcels"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    geom = Column(Geometry(geometry_type="POLYGON", srid=0, spatial_index=True), nullable=False)
    source = Column(String(20), default="ai", nullable=False)  # ai, manual, existing
    confidence = Column(Float, default=1.0, nullable=False)
    status = Column(String(20), default="auto", nullable=False)  # auto, edited, verified, rejected
    area = Column(Float, nullable=False)
    perimeter = Column(Float, nullable=False)
    attrs_json = Column(JSON, default=dict)
    version = Column(Integer, default=1, nullable=False)

    project = relationship("Project", back_populates="parcels")
    buildings = relationship("Building", back_populates="parcel")


class Building(Base):
    __tablename__ = "buildings"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    parcel_id = Column(String(36), ForeignKey("parcels.id", ondelete="SET NULL"), nullable=True)
    geom = Column(Geometry(geometry_type="POLYGON", srid=0, spatial_index=True), nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    height_m = Column(Float, nullable=True)

    project = relationship("Project", back_populates="buildings")
    parcel = relationship("Parcel", back_populates="buildings")


class Road(Base):
    __tablename__ = "roads"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    geom_poly = Column(Geometry(geometry_type="POLYGON", srid=0, spatial_index=True), nullable=True)
    geom_centerline = Column(Geometry(geometry_type="LINESTRING", srid=0, spatial_index=True), nullable=False)
    width_m = Column(Float, nullable=True)
    type = Column(String(50), default="local", nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)

    project = relationship("Project", back_populates="roads")


class LandUse(Base):
    __tablename__ = "landuse"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    geom = Column(Geometry(geometry_type="POLYGON", srid=0, spatial_index=True), nullable=False)
    landuse_class = Column("class", String(50), nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)

    project = relationship("Project", back_populates="landuses")


class ValidationIssue(Base):
    __tablename__ = "validation_issues"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    type = Column(String(50), nullable=False)  # overlap, gap, sliver, invalid, self_intersection, crosses_road, crosses_building, area_mismatch
    severity = Column(String(20), default="warning", nullable=False)  # error, warning, info
    geom = Column(Geometry(srid=0, spatial_index=True), nullable=True)
    feature_ids = Column(JSON, default=list)
    resolved = Column(Boolean, default=False, nullable=False)
    details = Column(JSON, default=dict)

    project = relationship("Project", back_populates="validation_issues")


class GTPoint(Base):
    __tablename__ = "gt_points"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    geom = Column(Geometry(geometry_type="POINT", srid=0, spatial_index=True), nullable=False)
    kind = Column(String(50), default="boundary_corner", nullable=False)
    note = Column(Text, nullable=True)
    photo_path = Column(Text, nullable=True)
    collected_by = Column(String(100), nullable=True)
    collected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    linked_feature_id = Column(String(36), nullable=True)

    project = relationship("Project", back_populates="gt_points")


class ControlPoint(Base):
    __tablename__ = "control_points"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    geom = Column(Geometry(geometry_type="POINT", srid=0, spatial_index=True), nullable=False)
    height = Column(Float, nullable=False)
    accuracy_cm = Column(Float, nullable=False)

    project = relationship("Project", back_populates="control_points")


class EditHistory(Base):
    __tablename__ = "edit_history"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    feature_table = Column(String(50), nullable=False)
    feature_id = Column(String(36), nullable=False)
    user = Column(String(100), default="system", nullable=False)
    action = Column(String(50), nullable=False)  # create, update, delete, split, merge, autofix
    before_geom = Column(JSON, nullable=True)
    after_geom = Column(JSON, nullable=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False)
