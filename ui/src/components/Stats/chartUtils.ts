const TICK_STEP_CANDIDATES = [1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 500, 1000, 2000, 2500, 5000];

export function niceTicks(max: number): { ticks: number[]; top: number } {
  if (max <= 0) return { ticks: [0, 1], top: 1 };
  const step = TICK_STEP_CANDIDATES.find((candidate) => candidate >= max / 4) ?? 10000;
  const top = Math.ceil(max / step) * step;
  const ticks: number[] = [];
  for (let value = 0; value <= top + 1e-9; value += step) ticks.push(value);
  return { ticks, top };
}
