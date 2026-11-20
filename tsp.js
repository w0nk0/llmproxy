// Exact TSP: Held-Karp bitmask DP, O(n^2 * 2^n) time, O(n * 2^n) memory.
// shortcut: dies past n ~ 20; swap in nearest-neighbour + 2-opt for big n.
function solveTSP(dist) {
  const n = dist.length;
  for (let i = 0; i < n; i++) {
    if (dist[i].length !== n) throw new Error("dist must be square");
    for (let j = 0; j < n; j++) {
      if (!(dist[i][j] >= 0)) throw new Error("dist must hold non-negative numbers");
    }
  }
  if (n === 0) return { cost: 0, tour: [] };

  const full = 1 << n;
  const dp = new Float64Array(full * n).fill(Infinity);
  const parent = new Int32Array(full * n).fill(-1);
  dp[n] = 0; // mask 1 = city 0 already visited, standing at city 0

  for (let mask = 1; mask < full; mask++) {
    if ((mask & 1) === 0) continue;
    for (let i = 0; i < n; i++) {
      const idx = mask * n + i;
      const cur = dp[idx];
      if (cur === Infinity || (mask & (1 << i)) === 0) continue;
      for (let j = 1; j < n; j++) {
        const bit = 1 << j;
        if (mask & bit) continue;
        const next = (mask | bit) * n + j;
        const alt = cur + dist[i][j];
        if (alt < dp[next]) {
          dp[next] = alt;
          parent[next] = i;
        }
      }
    }
  }

  const last = (full - 1) * n;
  let cost = Infinity;
  let end = -1;
  for (let i = 0; i < n; i++) {
    const total = dp[last + i] + dist[i][0];
    if (total < cost) {
      cost = total;
      end = i;
    }
  }

  const tour = [];
  let mask = full - 1;
  let i = end;
  while (i !== -1) {
    tour.push(i);
    const p = parent[mask * n + i];
    mask ^= 1 << i;
    i = p;
  }
  tour.reverse();
  return { cost, tour };
}

function tourCost(dist, tour) {
  if (tour.length === 0) return 0;
  let c = 0;
  for (let k = 0; k < tour.length; k++) {
    c += dist[tour[k]][tour[(k + 1) % tour.length]];
  }
  return c;
}

// reference implementation for the self-check: try every order, O(n!)
function bruteCost(dist) {
  const n = dist.length;
  if (n <= 1) return 0;
  const rest = Array.from({ length: n - 1 }, (_, k) => k + 1);
  let best = Infinity;
  const walk = (k) => {
    if (k === rest.length) {
      let c = 0;
      for (let t = 0; t + 1 < rest.length; t++) {
        c += dist[rest[t]][rest[t + 1]];
      }
      best = Math.min(best, dist[0][rest[0]] + c + dist[rest[rest.length - 1]][0]);
      return;
    }
    for (let i = k; i < rest.length; i++) {
      [rest[k], rest[i]] = [rest[i], rest[k]];
      walk(k + 1);
      [rest[k], rest[i]] = [rest[i], rest[k]];
    }
  };
  walk(0);
  return best;
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function main() {
  let s = 7;
  const rnd = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  for (let n = 1; n <= 8; n++) {
    for (const asymmetric of [false, true]) {
      const d = Array.from({ length: n }, () =>
        Array.from({ length: n }, () => Math.round(rnd() * 100)));
      for (let i = 0; i < n; i++) d[i][i] = 0;
      if (!asymmetric) {
        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) d[j][i] = d[i][j];
        }
      }
      const got = solveTSP(d);
      assert(got.cost === bruteCost(d), `cost not optimal at n=${n} asym=${asymmetric}`);
      assert(got.tour.length === n, `tour length wrong at n=${n}`);
      assert(new Set(got.tour).size === n, `tour repeats a city at n=${n}`);
      assert(tourCost(d, got.tour) === got.cost, `reported cost lies at n=${n}`);
    }
  }

  const pts = Array.from({ length: 12 }, () => [Math.round(rnd() * 100), Math.round(rnd() * 100)]);
  const d = Array.from(pts, ([x, y]) =>
    Array.from(pts, ([u, v]) => Math.hypot(x - u, y - v)));
  const got = solveTSP(d);
  console.log(`n=12 cost=${got.cost.toFixed(2)} tour=${got.tour.join(",")}`);
  console.log("ok");
}

main();
