/**
 * MINIMAL INPUT v2 — Frame-rate-independent movement, sprint, ESC unlock
 * 
 * Improvements over v1:
 *   - dt-based movement (no more speed variance at different FPS)
 *   - Sprint (hold Shift + W) = 2x speed
 *   - ESC releases pointer lock
 *   - Configurable mouse sensitivity
 *   - Vertical look: full -89° to +89° (was capped at ~86°)
 *   - Ground clamping prevents falling through floor
 *   - Arrow keys as WASD alternative
 */

export class MinimalInput {
  constructor(canvas, engine) {
    this.canvas = canvas;
    this.engine = engine;
    this.camera = { position: [128, 64, 200], target: [128, 64, 0] };
    this.mouse = { x: 0, y: 0, leftDown: false };
    this.keys = new Set();
    this.ray = { origin: null, direction: null };
    this.hitVoxel = null;
    this.hitNormal = null;
    this.pitch = -0.3;
    this.yaw = 0;

    // ── Tuning ──
    this.moveSpeed = 15.0;       // units/sec (was 3.0 * 60fps = 180/sec, now 15/sec steady)
    this.sprintMultiplier = 2.0;
    this.verticalSpeed = 12.0;
    this.mouseSensitivity = 0.002; // rad/pixel
    this.pitchMin = -1.55;       // ~-89°
    this.pitchMax = 1.55;        // ~+89°
    this.groundLevel = 0;        // floor Y — clamps player above this
    this.ceilingLevel = 255;     // max Y

    // ── Pointer lock state ──
    this._isLocked = false;

    this._setupListeners();
  }

  _setupListeners() {
    // ── Mouse look ──
    this.canvas.addEventListener('mousemove', (e) => {
      if (document.pointerLockElement === this.canvas) {
        this.yaw -= e.movementX * this.mouseSensitivity;
        this.pitch -= e.movementY * this.mouseSensitivity;
        this.pitch = Math.max(this.pitchMin, Math.min(this.pitchMax, this.pitch));
      }
      const rect = this.canvas.getBoundingClientRect();
      this.mouse.x = e.clientX - rect.left;
      this.mouse.y = e.clientY - rect.top;
    });

    // ── Click to lock ──
    this.canvas.addEventListener('click', () => {
      this.canvas.requestPointerLock();
    });

    // ── Pointer lock change ──
    document.addEventListener('pointerlockchange', () => {
      this._isLocked = document.pointerLockElement === this.canvas;
    });

    // ── Mining ──
    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0) {
        this.mouse.leftDown = true;
        this._handleClick();
      }
    });

    this.canvas.addEventListener('mouseup', (e) => {
      if (e.button === 0) this.mouse.leftDown = false;
    });

    // ── Keyboard (WASD + arrows + Shift + Space + Escape) ──
    window.addEventListener('keydown', (e) => {
      this.keys.add(e.code);

      // ESC: release pointer lock
      if (e.code === 'Escape' && this._isLocked) {
        document.exitPointerLock();
      }

      // Prevent browser defaults for game keys
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        e.preventDefault();
      }
    });

    window.addEventListener('keyup', (e) => this.keys.delete(e.code));

    // ── Tab/visibility: release lock when switching away ──
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && this._isLocked) {
        document.exitPointerLock();
      }
    });
  }

  _handleClick() {
    if (!this.hitVoxel) return;

    const [vx, vy, vz] = this.hitVoxel;

    if (this.engine.editManager && this.engine.editManager.submitEdit) {
      this.engine.editManager.submitEdit({
        center: [vx, vy, vz],
        radius: 2.0,
        materialType: 0,
        falloff: 0.3
      });
    }

    if (this.engine.audio && this.engine.audio.play) {
      this.engine.audio.play('mine', { volume: 0.5 + Math.random() * 0.3 });
    }
  }

  /**
   * Update camera position and raycast for this frame.
   * @param {number} dt — frame delta time in seconds
   */
  update(dt = 0.016) {
    // ── Build camera direction from yaw/pitch ──
    const dir = [
      Math.sin(this.yaw) * Math.cos(this.pitch),
      Math.sin(this.pitch),
      -Math.cos(this.yaw) * Math.cos(this.pitch)
    ];

    // ── Sprint: hold Shift while moving forward ──
    const sprinting = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight');
    const speed = this.moveSpeed * (sprinting ? this.sprintMultiplier : 1.0);

    // ── Forward/right vectors (horizontal plane only for WASD) ──
    const forward = [dir[0], 0, dir[2]];
    const fwdLen = Math.sqrt(forward[0] ** 2 + forward[2] ** 2) || 1;
    forward[0] /= fwdLen;
    forward[2] /= fwdLen;

    const right = [Math.cos(this.yaw), 0, Math.sin(this.yaw)];

    // ── WASD / Arrow movement (horizontal) ──
    let moved = false;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) {
      this.camera.position[0] += forward[0] * speed * dt;
      this.camera.position[2] += forward[2] * speed * dt;
      moved = true;
    }
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) {
      this.camera.position[0] -= forward[0] * speed * dt;
      this.camera.position[2] -= forward[2] * speed * dt;
      moved = true;
    }
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft')) {
      this.camera.position[0] -= right[0] * speed * dt;
      this.camera.position[2] -= right[2] * speed * dt;
      moved = true;
    }
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight')) {
      this.camera.position[0] += right[0] * speed * dt;
      this.camera.position[2] += right[2] * speed * dt;
      moved = true;
    }

    // ── Vertical movement ──
    if (this.keys.has('ShiftLeft') && !moved) {
      // Shift alone (no WASD) = descend
      this.camera.position[1] -= this.verticalSpeed * dt;
    }
    if (this.keys.has('Space')) {
      this.camera.position[1] += this.verticalSpeed * dt;
    }
    // Ctrl = descend (alternative)
    if (this.keys.has('ControlLeft') || this.keys.has('ControlRight')) {
      this.camera.position[1] -= this.verticalSpeed * dt;
    }

    // ── Ground/ceiling clamping ──
    if (this.camera.position[1] < this.groundLevel + 1) {
      this.camera.position[1] = this.groundLevel + 1;
    }
    if (this.camera.position[1] > this.ceilingLevel - 1) {
      this.camera.position[1] = this.ceilingLevel - 1;
    }

    // ── World boundary clamping ──
    this.camera.position[0] = Math.max(1, Math.min(254, this.camera.position[0]));
    this.camera.position[2] = Math.max(1, Math.min(254, this.camera.position[2]));

    // ── Update camera target ──
    this.camera.target = [
      this.camera.position[0] + dir[0],
      this.camera.position[1] + dir[1],
      this.camera.position[2] + dir[2]
    ];

    // ── Raycast ──
    this.ray.origin = [...this.camera.position];
    this.ray.direction = dir;
    this.hitVoxel = this._raymarchVoxels(this.ray.origin, this.ray.direction);

    return {
      camera: this.camera,
      ray: this.ray,
      hitVoxel: this.hitVoxel,
      isMining: this.mouse.leftDown,
      isLocked: this._isLocked,
      sprinting: sprinting && moved,
    };
  }

  _raymarchVoxels(origin, direction, maxSteps = 256) {
    let x = Math.floor(origin[0]);
    let y = Math.floor(origin[1]);
    let z = Math.floor(origin[2]);

    const dx = Math.sign(direction[0]);
    const dy = Math.sign(direction[1]);
    const dz = Math.sign(direction[2]);

    let tMaxX = dx === 0 ? Infinity : ((dx > 0 ? x + 1 : x) - origin[0]) / direction[0];
    let tMaxY = dy === 0 ? Infinity : ((dy > 0 ? y + 1 : y) - origin[1]) / direction[1];
    let tMaxZ = dz === 0 ? Infinity : ((dz > 0 ? z + 1 : z) - origin[2]) / direction[2];

    const tDeltaX = dx === 0 ? Infinity : Math.abs(1 / direction[0]);
    const tDeltaY = dy === 0 ? Infinity : Math.abs(1 / direction[1]);
    const tDeltaZ = dz === 0 ? Infinity : Math.abs(1 / direction[2]);

    for (let i = 0; i < maxSteps; i++) {
      if (x >= 0 && x < 256 && y >= 0 && y < 256 && z >= 0 && z < 256) {
        if (this._isVoxelSolid(x, y, z)) {
          return [x, y, z];
        }
      } else if (x < 0 || x >= 256 || z < 0 || z >= 256) {
        return null;
      }

      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) { x += dx; tMaxX += tDeltaX; }
        else { z += dz; tMaxZ += tDeltaZ; }
      } else {
        if (tMaxY < tMaxZ) { y += dy; tMaxY += tDeltaY; }
        else { z += dz; tMaxZ += tDeltaZ; }
      }
    }

    return null;
  }

  _isVoxelSolid(x, y, z) {
    if (x < 0 || x >= 256 || y < 0 || y >= 256 || z < 0 || z >= 256) return false;
    return x > 0 && x < 200 && y > 0 && y < 200 && z > 0 && z < 200;
  }
}
