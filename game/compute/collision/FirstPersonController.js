/**
 * FirstPersonController.js
 * 
 * Ties the 4 HyperPoly subsystems into a playable loop:
 * 1. PHYSICS    → Terrain evolves (gravity, erosion, water table)
 * 2. EXTRACTION → Mesh updates reflect physics (dual contouring)
 * 3. INPUT      → Mouse/keyboard drives camera & terrain editing
 * 4. COLLISION  → Sphere sweeps keep player grounded & prevent clipping
 * 
 * Framework-agnostic. Returns raw transforms for any renderer.
 */
export class FirstPersonController {
  constructor(canvas, engine, collisionSystem, options = {}) {
    this.canvas = canvas;
    this.engine = engine;
    this.collision = collisionSystem;
    this.cfg = {
      moveSpeed: 8.0,
      sprintMult: 1.8,
      jumpForce: 9.5,
      gravity: -24.0,
      mouseSens: 0.002,
      playerRadius: 0.45,
      playerHeight: 1.7,
      editRadius: 2.5,
      ...options
    };

    // State
    this.pos = new Float32Array([0, 60, 0]);
    this.vel = new Float32Array([0, 0, 0]);
    this.yaw = 0;
    this.pitch = 0;
    this.onGround = false;
    this.noclip = false;

    // Input
    this.keys = new Set();
    this.mouseDelta = { x: 0, y: 0 };
    this.pointerLocked = false;
    this.editPending = false;

    this._bindInput();
  }

  _bindInput() {
    this.canvas.addEventListener('click', () => {
      if (!this.pointerLocked) this.canvas.requestPointerLock();
    });

    document.addEventListener('pointerlockchange', () => {
      this.pointerLocked = document.pointerLockElement === this.canvas;
    });

    document.addEventListener('mousemove', (e) => {
      if (!this.pointerLocked) return;
      this.mouseDelta.x += e.movementX;
      this.mouseDelta.y += e.movementY;
    });

    window.addEventListener('keydown', (e) => {
      this.keys.add(e.code);
      if (e.code === 'KeyN') this.noclip = !this.noclip;
      if (e.code === 'Space' && this.onGround && !this.noclip) {
        this.vel[1] = this.cfg.jumpForce;
        this.onGround = false;
      }
    });

    window.addEventListener('keyup', (e) => this.keys.delete(e.code));

    this.canvas.addEventListener('mousedown', (e) => {
      if (e.button === 0 && this.pointerLocked) this.editPending = true;
    });
  }

  /**
   * Main update loop. Call once per frame.
   * @param {number} dt Delta time in seconds
   */
  update(dt) {
    this._handleLook(dt);
    this._handleMovement(dt);
    this._applyPhysics(dt);
    this._resolveCollision(dt);
    this._handleEdit();
    this._resetInputDelta();
  }

  _handleLook() {
    const sens = this.cfg.mouseSens;
    this.yaw -= this.mouseDelta.x * sens;
    this.pitch -= this.mouseDelta.y * sens;
    this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
  }

  _handleMovement(dt) {
    if (this.noclip) {
      const speed = this.cfg.moveSpeed * (this.keys.has('ShiftLeft') ? this.cfg.sprintMult : 1) * dt;
      const forward = [Math.sin(this.yaw) * Math.cos(this.pitch), 0, Math.cos(this.yaw) * Math.cos(this.pitch)];
      const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];

      const fLen = Math.hypot(forward[0], forward[2]);
      const rLen = Math.hypot(right[0], right[2]);
      forward[0] /= fLen; forward[2] /= fLen;
      right[0] /= rLen; right[2] /= rLen;

      let dx = 0, dz = 0;
      if (this.keys.has('KeyW')) { dx += forward[0]; dz += forward[2]; }
      if (this.keys.has('KeyS')) { dx -= forward[0]; dz -= forward[2]; }
      if (this.keys.has('KeyA')) { dx -= right[0]; dz -= right[2]; }
      if (this.keys.has('KeyD')) { dx += right[0]; dz += right[2]; }
      if (this.keys.has('Space')) this.pos[1] += speed;
      if (this.keys.has('ShiftLeft') && !this.keys.has('Space')) this.pos[1] -= speed;

      const len = Math.hypot(dx, dz);
      if (len > 0) { dx /= len; dz /= len; }
      this.pos[0] += dx * speed;
      this.pos[2] += dz * speed;
      this.vel[1] = 0;
      return;
    }

    const speed = this.cfg.moveSpeed * (this.keys.has('ShiftLeft') ? this.cfg.sprintMult : 1);
    const forward = [Math.sin(this.yaw), 0, Math.cos(this.yaw)];
    const right = [Math.cos(this.yaw), 0, -Math.sin(this.yaw)];

    let mx = 0, mz = 0;
    if (this.keys.has('KeyW')) { mx += forward[0]; mz += forward[2]; }
    if (this.keys.has('KeyS')) { mx -= forward[0]; mz -= forward[2]; }
    if (this.keys.has('KeyA')) { mx -= right[0]; mz -= right[2]; }
    if (this.keys.has('KeyD')) { mx += right[0]; mz += right[2]; }

    const len = Math.hypot(mx, mz);
    if (len > 0) { mx /= len; mz /= len; }
    this.vel[0] = mx * speed;
    this.vel[2] = mz * speed;
  }

  _applyPhysics(dt) {
    if (this.noclip) return;
    this.vel[1] += this.cfg.gravity * dt;
    this.vel[1] = Math.max(this.vel[1], -40);
  }

  _resolveCollision(dt) {
    if (this.noclip) { this.onGround = false; return; }

    const radius = this.cfg.playerRadius;
    const eyeHeight = this.cfg.playerHeight;

    // Ground collision
    const feetPos = [this.pos[0], this.pos[1] - eyeHeight + radius, this.pos[2]];
    const groundHit = this.collision.sphereCast(feetPos, radius, [0, -2.0, 0]);

    if (groundHit.hit && groundHit.distance < radius + 0.1) {
      this.onGround = true;
      this.pos[1] = groundHit.point[1] + eyeHeight + 0.02;
      this.vel[1] = Math.max(0, this.vel[1]);
    } else {
      this.onGround = false;
    }

    // Wall collision
    if (Math.hypot(this.vel[0], this.vel[2]) > 0.01) {
      const moveDir = [this.vel[0], 0, this.vel[2]];
      const len = Math.hypot(...moveDir);
      const normDir = [moveDir[0]/len, 0, moveDir[2]/len];
      const sweepDist = len * dt + radius;

      const center = [this.pos[0], this.pos[1] - eyeHeight/2, this.pos[2]];
      const wallHit = this.collision.sphereCast(center, radius, [normDir[0]*sweepDist, 0, normDir[2]*sweepDist]);

      if (wallHit.hit) {
        const pushAmount = radius - wallHit.distance + 0.01;
        this.pos[0] += wallHit.normal[0] * pushAmount;
        this.pos[2] += wallHit.normal[2] * pushAmount;
        this.vel[0] = 0;
        this.vel[2] = 0;
      }
    }

    this.pos[0] += this.vel[0] * dt;
    this.pos[2] += this.vel[2] * dt;

    if (!this.onGround) {
      this.pos[1] += this.vel[1] * dt;
    }

    if (this.pos[1] < -50) { this.pos[1] = 100; this.vel[1] = 0; }
  }

  _handleEdit() {
    if (!this.editPending) return;
    this.editPending = false;

    const forward = [
      Math.sin(this.yaw) * Math.cos(this.pitch),
      Math.sin(this.pitch),
      Math.cos(this.yaw) * Math.cos(this.pitch)
    ];
    const origin = [this.pos[0], this.pos[1] - this.cfg.playerHeight + 0.2, this.pos[2]];
    const hit = this.collision.raycast(origin, forward, 32.0);

    if (hit.hit) {
      this.engine.edit({
        center: hit.point,
        radius: this.cfg.editRadius,
        materialType: 'air',
        falloff: 0.4
      });
    }
  }

  _resetInputDelta() {
    this.mouseDelta.x = 0;
    this.mouseDelta.y = 0;
  }

  getCameraState() {
    return {
      position: this.pos,
      yaw: this.yaw,
      pitch: this.pitch,
      fov: 70 * Math.PI / 180
    };
  }

  getAxes() {
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    return {
      forward: [sy * cp, sp, cy * cp],
      right: [cy, 0, -sy],
      up: [0, 1, 0]
    };
  }

  destroy() {
    if (document.pointerLockElement === this.canvas) document.exitPointerLock();
  }
}
