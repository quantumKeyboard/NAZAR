/* eslint-disable */
// GENERATED FILE - do not edit. Source: packages/contracts/python (Pydantic).
// Regenerate with `pnpm contracts:generate`.

/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ScaleStatus".
 */
export type ScaleStatus = 'unknown' | 'relative' | 'metric_estimated' | 'metric_sensor';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "FrameType".
 */
export type FrameType = 'local_reconstruction' | 'enu' | 'ecef' | 'geographic' | 'projected';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ArtifactKind".
 */
export type ArtifactKind =
  | 'keyframes'
  | 'point_cloud_preview'
  | 'point_cloud'
  | 'camera_trajectory'
  | 'depth_maps'
  | 'dynamic_masks'
  | 'gaussian_scene'
  | 'mesh'
  | 'report'
  | 'export';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AssetKind".
 */
export type AssetKind = 'video' | 'telemetry';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AssetStatus".
 */
export type AssetStatus =
  'uploading' | 'validating' | 'ready' | 'rejected' | 'failed' | 'cancelled';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "EventType".
 */
export type EventType =
  | 'job.created'
  | 'job.queued'
  | 'stage.started'
  | 'stage.progress'
  | 'stage.completed'
  | 'stage.skipped'
  | 'job.estimate'
  | 'warning'
  | 'artifact.available'
  | 'preview.updated'
  | 'job.completed'
  | 'job.failed'
  | 'cancel.requested'
  | 'job.cancelled';
/**
 * Which worker location runs this job.
 */
export type ComputeTarget = 'local' | 'college';
/**
 * Which stage implementations a job uses.
 *
 * - `test`: deterministic test pipeline. Emits small real-format fixture artifacts through
 *   the real API and event contracts. Never presented as a reconstruction of the input.
 * - `standard`: real stages (Fast3R on the GPU when the worker has it, else the CPU
 *   classical-geometry baseline).
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PipelineMode".
 */
export type PipelineMode = 'test' | 'standard';
/**
 * Processing stages. Each value is also the job status while that stage runs.
 *
 * Moving objects are masked *before* geometry, so the masks keep them out of the pose
 * estimate as well as out of the points.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageName".
 */
export type StageName =
  | 'preprocessing'
  | 'dynamic_analysis'
  | 'geometry'
  | 'depth_priors'
  | 'sensor_optimization'
  | 'dense_reconstruction'
  | 'surface_processing'
  | 'analysis'
  | 'exporting';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageRunState".
 */
export type StageRunState =
  'pending' | 'running' | 'completed' | 'skipped' | 'failed' | 'cancelled';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobStatus".
 */
export type JobStatus =
  | 'created'
  | 'validating'
  | 'queued'
  | 'preprocessing'
  | 'dynamic_analysis'
  | 'geometry'
  | 'depth_priors'
  | 'sensor_optimization'
  | 'dense_reconstruction'
  | 'surface_processing'
  | 'analysis'
  | 'exporting'
  | 'completed'
  | 'failed'
  | 'cancel_requested'
  | 'cancelled';
export type JobEvent =
  | JobCreated
  | JobQueued
  | StageStarted
  | StageProgress
  | StageCompleted
  | StageSkipped
  | JobEstimate
  | WarningRaised
  | ArtifactAvailable
  | PreviewUpdated
  | JobCompleted
  | JobFailed
  | CancelRequested
  | JobCancelled;
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "KeyframeReason".
 */
export type KeyframeReason = 'start' | 'novelty' | 'max_gap' | 'end' | 'sampled';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "FrameFlag".
 */
export type FrameFlag = 'blurry' | 'dark' | 'overexposed' | 'duplicate';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WarningSeverity".
 */
export type WarningSeverity = 'info' | 'warning' | 'blocking';
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "CheckStatus".
 */
export type CheckStatus = 'ok' | 'degraded' | 'unavailable' | 'not_configured' | 'not_implemented';
/**
 * What a worker's standard pipeline would use for geometry right now.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "GeometryBackend".
 */
export type GeometryBackend = 'fast3r' | 'cpu-baseline';
/**
 * Where a job's GPU work runs. Each worker declares its location and only leases jobs
 * aimed at it (the API never moves a job between locations on its own).
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ComputeTarget".
 */
export type ComputeTarget1 = 'local' | 'college';
/**
 * Altitudes keep their meaning: height above take-off is not sea-level or ellipsoidal.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AltitudeKind".
 */
export type AltitudeKind = 'relative_to_takeoff' | 'absolute_unspecified_datum' | 'none';
/**
 * Where a job's GPU work runs. Each worker declares its location and only leases jobs
 * aimed at it (the API never moves a job between locations on its own).
 */
export type ComputeTarget2 = 'local' | 'college';

/**
 * Bundle that pulls every public contract into one schema document.
 */
export interface NazarContracts {
  artifact_create: ArtifactCreate;
  artifact_created: ArtifactCreated;
  artifact_manifest: ArtifactManifest;
  asset: Asset;
  asset_create: AssetCreate;
  auth_state: AuthState;
  credentials: Credentials;
  event_type: EventType;
  health: HealthResponse;
  job: Job;
  job_event: JobEvent;
  job_status: JobStatus;
  mission: Mission;
  mission_create: MissionCreate;
  preflight_report: PreflightReport;
  readiness: ReadinessResponse;
  stage_name: StageName;
  telemetry_summary: TelemetrySummary;
  worker_event_ack: WorkerEventAck;
  worker_event_batch: WorkerEventBatch;
  worker_heartbeat: WorkerHeartbeat;
  worker_lease: WorkerLease;
  worker_lease_request: WorkerLeaseRequest;
}
/**
 * Declared before upload; bytes are verified against `byte_size` and the checksum.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ArtifactCreate".
 */
export interface ArtifactCreate {
  bounds?: Bounds3D | null;
  byte_size: number;
  checksum_sha256: string;
  /**
   * False for previews that a later revision may supersede.
   */
  final?: boolean;
  frame?: CoordinateFrame | null;
  /**
   * Idempotency key.
   */
  key: string;
  kind: ArtifactKind;
  lod?: number | null;
  media_type: string;
  point_count?: number | null;
  provenance: Provenance;
  source_frames?: FrameRange | null;
  transform_to_frame?:
    | [
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number
      ]
    | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Bounds3D".
 */
export interface Bounds3D {
  /**
   * @minItems 3
   * @maxItems 3
   */
  max: [number, number, number];
  /**
   * @minItems 3
   * @maxItems 3
   */
  min: [number, number, number];
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "CoordinateFrame".
 */
export interface CoordinateFrame {
  /**
   * Axis convention of stored coordinates.
   */
  axes?: string;
  /**
   * EPSG/WKT identifier when georeferenced.
   */
  crs?: string | null;
  /**
   * Metres per coordinate unit when scale comes from a sensor alignment while coordinates stay in the local frame; null when scale is unknown or units are metres.
   */
  metres_per_unit?: number | null;
  scale_status: ScaleStatus;
  type: FrameType;
  /**
   * Linear units of coordinates, e.g. 'metre' or 'unitless'.
   */
  units: string;
  vertical_datum?: string | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Provenance".
 */
export interface Provenance {
  config_hash?: string | null;
  model?: string | null;
  model_version?: string | null;
  /**
   * Stage adapter that produced the artifact.
   */
  producer: string;
  /**
   * True when produced by the deterministic test pipeline, not a real model.
   */
  test_pipeline: boolean;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "FrameRange".
 */
export interface FrameRange {
  count: number;
  first_index: number;
  last_index: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ArtifactCreated".
 */
export interface ArtifactCreated {
  id: string;
  revision: number;
  /**
   * True when the bytes were already received (retry).
   */
  uploaded: boolean;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ArtifactManifest".
 */
export interface ArtifactManifest {
  bounds?: Bounds3D | null;
  byte_size: number;
  checksum_sha256: string;
  created_at: string;
  frame?: CoordinateFrame | null;
  id: string;
  job_id: string;
  kind: ArtifactKind;
  lod?: number | null;
  media_type: string;
  point_count?: number | null;
  provenance: Provenance;
  /**
   * Monotonic per (job, kind); stale revisions are ignored.
   */
  revision: number;
  source_frames?: FrameRange | null;
  /**
   * Column-major 4x4 from artifact-local coords to `frame`.
   */
  transform_to_frame?:
    | [
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number,
        number
      ]
    | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Asset".
 */
export interface Asset {
  byte_size: number;
  created_at: string;
  error: string | null;
  filename: string;
  id: string;
  kind: AssetKind;
  mission_id: string;
  progress: ProcessingProgress | null;
  received_bytes: number;
  sha256: string | null;
  status: AssetStatus;
  updated_at: string;
}
/**
 * Server-side processing phase. Counts are null when the work is not countable.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ProcessingProgress".
 */
export interface ProcessingProgress {
  completed?: number | null;
  phase: string;
  total?: number | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AssetCreate".
 */
export interface AssetCreate {
  byte_size: number;
  filename: string;
  kind: AssetKind;
}
/**
 * `setup_required` is true only while no operator account exists.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AuthState".
 */
export interface AuthState {
  setup_required: boolean;
  user: UserPublic | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "UserPublic".
 */
export interface UserPublic {
  email: string;
  id: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Credentials".
 */
export interface Credentials {
  email: string;
  password: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "HealthResponse".
 */
export interface HealthResponse {
  environment: string;
  service: string;
  status: 'ok';
  time: string;
  version: string;
}
/**
 * Current snapshot of a job, rebuilt from its event log.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Job".
 */
export interface Job {
  /**
   * Newest revision of each artifact.
   */
  artifacts: ArtifactManifest[];
  attempt: number;
  compute?: ComputeTarget;
  created_at: string;
  error?: string | null;
  /**
   * The worker's time forecast, once it has leased the job.
   */
  estimate?: JobEstimateData | null;
  id: string;
  /**
   * Cursor of the newest event; resume after it.
   */
  last_seq: number;
  mission_id: string;
  pipeline: PipelineMode;
  retryable?: boolean;
  /**
   * Frames used (standard pipeline); None for the test pipeline.
   */
  sampling?: SamplingInfo | null;
  stages: StageState[];
  status: JobStatus;
  updated_at: string;
  warnings: WarningData[];
}
/**
 * The worker's forecast of how long each stage will take, sent once the job is leased.
 * Clients combine it with the stages' real progress to show the time remaining.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobEstimateData".
 */
export interface JobEstimateData {
  /**
   * Where the numbers come from, in words (shown to the user).
   */
  basis: string;
  /**
   * Frames the estimate was made for.
   */
  frames: number;
  /**
   * Estimated seconds per stage; 0 for a stage already known to be skipped.
   */
  stages: {
    [k: string]: number;
  };
  total_s: number;
}
/**
 * Which frames of the video a job uses: chosen at the requested rate (or automatically),
 * one per time window (the sharpest usable frame), and lowered to the location's frame cap
 * if needed.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "SamplingInfo".
 */
export interface SamplingInfo {
  /**
   * True when the frame count was chosen automatically from the video length.
   */
  auto?: boolean;
  /**
   * Most frames this job's compute location allows.
   */
  cap: number;
  /**
   * True when the request needed more frames than the cap.
   */
  clamped: boolean;
  /**
   * Frames actually used per second of video.
   */
  fps_effective: number;
  fps_requested: number;
  frames: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageState".
 */
export interface StageState {
  /**
   * GPU memory budget it ran under.
   */
  budget_mib?: number | null;
  duration_s?: number | null;
  /**
   * Peak GPU memory it used.
   */
  peak_vram_mib?: number | null;
  progress?: StageProgressData | null;
  /**
   * Why it was skipped or failed.
   */
  reason?: string | null;
  stage: StageName;
  /**
   * When the stage started (server clock).
   */
  started_at?: string | null;
  state: StageRunState;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageProgressData".
 */
export interface StageProgressData {
  /**
   * Units of real work done. Null when not measurable.
   */
  completed?: number | null;
  message: string;
  /**
   * Total units. Null means indeterminate progress.
   */
  total?: number | null;
  /**
   * e.g. 'frames', 'keyframes', 'iterations'
   */
  unit?: string | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WarningData".
 */
export interface WarningData {
  code: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobCreated".
 */
export interface JobCreated {
  data: JobCreatedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.created';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobCreatedData".
 */
export interface JobCreatedData {
  mission_id: string;
  test_pipeline: boolean;
}
/**
 * The job went back to the queue (user retry, or a worker stopped renewing its lease).
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobQueued".
 */
export interface JobQueued {
  data: JobQueuedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.queued';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobQueuedData".
 */
export interface JobQueuedData {
  attempt: number;
  /**
   * Why the job (re)entered the queue, e.g. retry or lost worker.
   */
  reason: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageStarted".
 */
export interface StageStarted {
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'stage.started';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageProgress".
 */
export interface StageProgress {
  data: StageProgressData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'stage.progress';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageCompleted".
 */
export interface StageCompleted {
  data: StageCompletedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'stage.completed';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageCompletedData".
 */
export interface StageCompletedData {
  /**
   * The GPU memory budget it ran under (0: unlimited).
   */
  budget_mib?: number | null;
  duration_s: number;
  /**
   * Peak GPU memory this stage's models used, if any.
   */
  peak_vram_mib?: number | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageSkipped".
 */
export interface StageSkipped {
  data: StageSkippedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'stage.skipped';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "StageSkippedData".
 */
export interface StageSkippedData {
  reason: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobEstimate".
 */
export interface JobEstimate {
  data: JobEstimateData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.estimate';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WarningRaised".
 */
export interface WarningRaised {
  data: WarningData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'warning';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ArtifactAvailable".
 */
export interface ArtifactAvailable {
  data: ArtifactManifest;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'artifact.available';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PreviewUpdated".
 */
export interface PreviewUpdated {
  data: PreviewUpdatedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'preview.updated';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PreviewUpdatedData".
 */
export interface PreviewUpdatedData {
  artifact_id: string;
  revision: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobCompleted".
 */
export interface JobCompleted {
  data: JobCompletedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.completed';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobCompletedData".
 */
export interface JobCompletedData {
  artifact_ids: string[];
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobFailed".
 */
export interface JobFailed {
  data: JobFailedData;
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.failed';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobFailedData".
 */
export interface JobFailedData {
  /**
   * Human-readable, actionable failure reason.
   */
  reason: string;
  retryable: boolean;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "CancelRequested".
 */
export interface CancelRequested {
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'cancel.requested';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "JobCancelled".
 */
export interface JobCancelled {
  job_id: string;
  schema_version?: 1;
  seq: number;
  stage?: StageName | null;
  status: JobStatus;
  ts: string;
  type: 'job.cancelled';
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Mission".
 */
export interface Mission {
  created_at: string;
  id: string;
  name: string;
  telemetry?: Asset | null;
  video: Asset | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "MissionCreate".
 */
export interface MissionCreate {
  name: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PreflightReport".
 */
export interface PreflightReport {
  analysis: AnalysisInfo;
  asset_id: string;
  generated_at: string;
  keyframes: Keyframe[];
  probe: VideoProbe;
  schema_version?: 1;
  series: FrameSeries;
  settings: {
    [k: string]: number | string;
  };
  source_sha256: string;
  summary: PreflightSummary;
  telemetry_present: boolean;
  warnings: PreflightWarning[];
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "AnalysisInfo".
 */
export interface AnalysisInfo {
  decode_errors: number;
  frames_decoded: number;
  height: number;
  processing_s: number;
  timestamps: 'stream' | 'estimated';
  width: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "Keyframe".
 */
export interface Keyframe {
  image: number;
  index: number;
  quality: number;
  reason: KeyframeReason;
  t: number;
}
/**
 * Facts read from the container/stream by ffprobe. `display_*` include rotation.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "VideoProbe".
 */
export interface VideoProbe {
  average_fps: number | null;
  bit_rate: number | null;
  codec: string;
  container: string;
  display_height: number;
  display_width: number;
  duration_s: number;
  frame_count: number | null;
  has_audio: boolean;
  height: number;
  nominal_fps: number | null;
  pix_fmt: string | null;
  rotation_deg: number;
  variable_frame_rate: boolean;
  width: number;
}
/**
 * Per-frame metrics, column-oriented (one entry per decoded frame, in display order).
 *
 * - `t`: presentation time in seconds from the stream's first frame.
 * - `sharpness`: variance of the Laplacian at analysis resolution (higher = sharper).
 * - `brightness`: mean luma, 0-1.
 * - `change`: mean absolute luma difference to the previous frame, 0-1.
 * - `motion_px`: estimated global image shift to the previous frame, in analysis pixels.
 * - `flags`: per-frame quality flags.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "FrameSeries".
 */
export interface FrameSeries {
  brightness: number[];
  change: number[];
  flags: FrameFlag[][];
  index: number[];
  motion_px: number[];
  sharpness: number[];
  t: number[];
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PreflightSummary".
 */
export interface PreflightSummary {
  blurry_frames: number;
  dark_frames: number;
  duplicate_frames: number;
  median_sharpness: number;
  overexposed_frames: number;
  usable_frames: number;
  viewpoint_change: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "PreflightWarning".
 */
export interface PreflightWarning {
  code: string;
  message: string;
  severity: WarningSeverity;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ReadinessResponse".
 */
export interface ReadinessResponse {
  checks: ReadinessCheck[];
  ready: boolean;
  /**
   * One entry per compute location.
   */
  workers?: WorkerPresence[];
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ReadinessCheck".
 */
export interface ReadinessCheck {
  detail: string;
  name: string;
  required: boolean;
  status: CheckStatus;
}
/**
 * Whether a worker at this location polled the API recently (presence, not a secret).
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerPresence".
 */
export interface WorkerPresence {
  /**
   * Geometry backend it reported for standard jobs (None: not reported).
   */
  geometry: GeometryBackend | null;
  /**
   * Seconds since its last call; None: never.
   */
  last_seen_s: number | null;
  location: ComputeTarget1;
  /**
   * Models it can run right now (None: not reported).
   */
  models?: ModelReadiness[] | null;
  online: boolean;
  /**
   * Its time per frame, for estimates before a job (None: unknown).
   */
  time_model?: TimeModel | null;
}
/**
 * One model a worker can run: installed or not, and why not.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "ModelReadiness".
 */
export interface ModelReadiness {
  detail: string;
  name: string;
  ready: boolean;
  stage: StageName;
}
/**
 * A worker's total job time as a straight line in the frame count, so the UI can quote an
 * estimate before a job exists: `fixed_s + per_frame_s * frames` (no telemetry assumed).
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "TimeModel".
 */
export interface TimeModel {
  basis: string;
  fixed_s: number;
  per_frame_s: number;
}
/**
 * Computed from the normalised track. Times are seconds since the video started.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "TelemetrySummary".
 */
export interface TelemetrySummary {
  altitude_kind: AltitudeKind;
  /**
   * Rejected samples by reason.
   */
  dropped: {
    [k: string]: number;
  };
  end_s: number;
  /**
   * Plain-language notes on limits of this telemetry.
   */
  findings: string[];
  /**
   * dji_srt, airdata_csv or csv.
   */
  format: string;
  /**
   * Pauses longer than 1.5 s; never interpolated.
   */
  gaps: TimeGap[];
  height_range_m: number | null;
  /**
   * Largest horizontal distance from the first position (via WGS84 ENU).
   */
  horizontal_extent_m: number;
  /**
   * Median sample rate.
   */
  rate_hz: number;
  samples: number;
  start_s: number;
  /**
   * Fraction of the video's duration covered by samples outside gaps.
   */
  video_coverage: number | null;
  /**
   * From the video's preflight; null until the video is validated.
   */
  video_duration_s: number | null;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "TimeGap".
 */
export interface TimeGap {
  end_s: number;
  start_s: number;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerEventAck".
 */
export interface WorkerEventAck {
  cancel_requested: boolean;
  last_seq: number;
  status: JobStatus;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerEventBatch".
 */
export interface WorkerEventBatch {
  /**
   * @minItems 1
   * @maxItems 100
   */
  events: [WorkerEvent, ...WorkerEvent[]];
}
/**
 * An event reported by the worker. The API assigns `seq`, `ts` and `status`.
 *
 * `key` makes delivery idempotent: re-sending an event with a key the job already has is a
 * no-op, so the worker can safely retry after a dropped connection.
 *
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerEvent".
 */
export interface WorkerEvent {
  data?: {
    [k: string]: unknown;
  } | null;
  key: string;
  stage?: StageName | null;
  type: EventType;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerHeartbeat".
 */
export interface WorkerHeartbeat {
  cancel_requested: boolean;
  lease_expires_at: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerLease".
 */
export interface WorkerLease {
  attempt: number;
  job_id: string;
  keyframe_count: number;
  lease_expires_at: string;
  /**
   * Send as `X-Nazar-Lease` on every call for this job.
   */
  lease_id: string;
  mission_id: string;
  pipeline: PipelineMode;
  sampling?: SamplingInfo | null;
  /**
   * Accepted telemetry pinned when the job was created; fetch it from `inputs/telemetry`.
   */
  telemetry_asset_id?: string | null;
  video_asset_id: string;
  video_sha256: string;
}
/**
 * This interface was referenced by `NazarContracts`'s JSON-Schema
 * via the `definition` "WorkerLeaseRequest".
 */
export interface WorkerLeaseRequest {
  /**
   * Geometry backend for standard jobs on this worker (absent: not reported).
   */
  geometry?: GeometryBackend | null;
  /**
   * Lease only this queued job (offline bundle import); absent: oldest first.
   */
  job_id?: string | null;
  location?: ComputeTarget2;
  /**
   * Models this worker can run right now (absent: not reported).
   */
  models?: ModelReadiness[] | null;
  /**
   * @minItems 1
   */
  modes: [PipelineMode, ...PipelineMode[]];
  /**
   * How long this worker takes per frame (absent: not reported).
   */
  time_model?: TimeModel | null;
  worker_id: string;
}
