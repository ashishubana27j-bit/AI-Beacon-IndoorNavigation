// ─────────────────────────────────────────────────────────
//  kalman.js  — 1-D Kalman filter
//  R = measurement noise  (low → trust beacon fast)
//  Q = process noise      (high → allow quick movement)
// ─────────────────────────────────────────────────────────
export default class KalmanFilter {
  constructor({ R = 1, Q = 1 } = {}) {
    this.R   = R;
    this.Q   = Q;
    this.A   = 1;
    this.C   = 1;
    this.x   = null;
    this.cov = NaN;
  }

  filter(z) {
    if (this.x === null) {
      this.x   = z / this.C;
      this.cov = (1 / this.C) * this.Q * (1 / this.C);
    } else {
      const predX   = this.A * this.x;
      const predCov = this.A * this.cov * this.A + this.Q;
      const K       = predCov * this.C / (this.C * predCov * this.C + this.R);
      this.x        = predX + K * (z - this.C * predX);
      this.cov      = predCov - K * this.C * predCov;
    }
    return this.x;
  }

  reset() { this.x = null; this.cov = NaN; }
}
