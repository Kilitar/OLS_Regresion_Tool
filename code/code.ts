import React, { useState, useMemo } from 'react';

interface Point {
  x: number;
  y: number;
}

const DEFAULT_POINTS: Point[] = [
  { x: 1, y: 2.2 },
  { x: 2, y: 2.8 },
  { x: 3, y: 4.5 },
  { x: 4, y: 3.8 },
  { x: 5, y: 5.8 },
  { x: 6, y: 5.2 },
  { x: 7, y: 7.1 },
];

export const OLSVisualizer: React.FC = () => {
  // Rozsahy os grafu
  const xMin = 0;
  const xMax = 8;
  const yMin = 0;
  const yMax = 9;

  // Analytický výpoèet optimální OLS pøímky
  const optimalOLS = useMemo(() => {
    const n = DEFAULT_POINTS.length;
    const sumX = DEFAULT_POINTS.reduce((acc, p) => acc + p.x, 0);
    const sumY = DEFAULT_POINTS.reduce((acc, p) => acc + p.y, 0);
    const meanX = sumX / n;
    const meanY = sumY / n;

    const numerator = DEFAULT_POINTS.reduce(
      (acc, p) => acc + (p.x - meanX) * (p.y - meanY),
      0
    );
    const denominator = DEFAULT_POINTS.reduce(
      (acc, p) => acc + Math.pow(p.x - meanX, 2),
      0
    );

    const slope = numerator / denominator;
    const intercept = meanY - slope * meanX;

    const minSSE = DEFAULT_POINTS.reduce((acc, p) => {
      const yPred = slope * p.x + intercept;
      return acc + Math.pow(p.y - yPred, 2);
    }, 0);

    return { slope, intercept, minSSE };
  }, []);

  const [slope, setSlope] = useState<number>(0.8);
  const [intercept, setIntercept] = useState<number>(1.2);
  const [showSquares, setShowSquares] = useState<boolean>(true);

  // Pøepoèet aktuálního souètu ètvercù (SSE)
  const currentSSE = useMemo(() => {
    return DEFAULT_POINTS.reduce((acc, p) => {
      const yPred = slope * p.x + intercept;
      return acc + Math.pow(p.y - yPred, 2);
    }, 0);
  }, [slope, intercept]);

  // Nastavení parametrù na analytické minimum OLS
  const handleSnapToOptimal = () => {
    setSlope(parseFloat(optimalOLS.slope.toFixed(3)));
    setIntercept(parseFloat(optimalOLS.intercept.toFixed(3)));
  };

  // Rozmìry SVG plátna
  const width = 600;
  const height = 400;
  const margin = { top: 20, right: 30, bottom: 40, left: 50 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Transformace datových souøadnic na pixely
  const scaleX = (x: number) => margin.left + ((x - xMin) / (xMax - xMin)) * innerWidth;
  const scaleY = (y: number) => margin.top + innerHeight - ((y - yMin) / (yMax - yMin)) * innerHeight;

  // Mìøítko pixelù na jednotku pro výpoèet geometrického ètverce
  const pxPerUnitX = innerWidth / (xMax - xMin);
  const pxPerUnitY = innerHeight / (yMax - yMin);

  return (
    <div className="w-full max-w-2xl bg-white border border-slate-200 rounded-lg shadow-sm p-6 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">Metoda nejmenších ètvercù (OLS)</h2>
          <p className="text-sm text-slate-500">
            Minimalizace souètu ètvercù vertikálních odchylek
          </p>
        </div>
        <div className="flex items-center gap-4 text-sm font-mono">
          <div className="bg-slate-50 px-3 py-1.5 rounded border border-slate-200">
            SSE: <span className="font-bold text-slate-800">{currentSSE.toFixed(3)}</span>
          </div>
          <div className="bg-emerald-50 px-3 py-1.5 rounded border border-emerald-200 text-emerald-800">
            Min OLS: <span className="font-bold">{optimalOLS.minSSE.toFixed(3)}</span>
          </div>
        </div>
      </div>

      {/* SVG Graf */}
      <div className="flex justify-center bg-slate-50 rounded-lg p-2 border border-slate-100">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          {/* Møížka a osy */}
          {Array.from({ length: xMax - xMin + 1 }).map((_, i) => {
            const xVal = xMin + i;
            return (
              <g key={`x-grid-${xVal}`}>
                <line
                  x1={scaleX(xVal)}
                  y1={scaleY(yMin)}
                  x2={scaleX(xVal)}
                  y2={scaleY(yMax)}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text
                  x={scaleX(xVal)}
                  y={scaleY(yMin) + 20}
                  fontSize="12"
                  textAnchor="middle"
                  fill="#64748b"
                >
                  {xVal}
                </text>
              </g>
            );
          })}

          {Array.from({ length: yMax - yMin + 1 }).map((_, i) => {
            const yVal = yMin + i;
            return (
              <g key={`y-grid-${yVal}`}>
                <line
                  x1={scaleX(xMin)}
                  y1={scaleY(yVal)}
                  x2={scaleX(xMax)}
                  y2={scaleY(yVal)}
                  stroke="#e2e8f0"
                  strokeWidth="1"
                />
                <text
                  x={scaleX(xMin) - 10}
                  y={scaleY(yVal) + 4}
                  fontSize="12"
                  textAnchor="end"
                  fill="#64748b"
                >
                  {yVal}
                </text>
              </g>
            );
          })}

          {/* Obyèejné osy */}
          <line
            x1={scaleX(xMin)}
            y1={scaleY(yMin)}
            x2={scaleX(xMax)}
            y2={scaleY(yMin)}
            stroke="#475569"
            strokeWidth="1.5"
          />
          <line
            x1={scaleX(xMin)}
            y1={scaleY(yMin)}
            x2={scaleX(xMin)}
            y2={scaleY(yMax)}
            stroke="#475569"
            strokeWidth="1.5"
          />

          {/* Geometrické ètverce a rezidua */}
          {DEFAULT_POINTS.map((pt, idx) => {
            const yPred = slope * pt.x + intercept;
            const diff = pt.y - yPred; // reziduum v datových jednotkách

            const pxX = scaleX(pt.x);
            const pxY = scaleY(pt.y);
            const pxYPred = scaleY(yPred);

            const sidePxY = Math.abs(pxY - pxYPred);
            // Zachování pomìru stran ètverce v prostoru grafu
            const sidePxX = Math.abs(diff) * pxPerUnitX;

            // Ètverec kreslíme smìrem doprava od reziduální úseèky
            const rectX = pxX;
            const rectY = Math.min(pxY, pxYPred);

            return (
              <g key={`residual-${idx}`}>
                {showSquares && (
                  <rect
                    x={rectX}
                    y={rectY}
                    width={sidePxX}
                    height={sidePxY}
                    fill="#f87171"
                    fillOpacity="0.25"
                    stroke="#ef4444"
                    strokeWidth="1"
                    strokeDasharray="2 2"
                  />
                )}
                {/* Svislá úseèka rezidua */}
                <line
                  x1={pxX}
                  y1={pxY}
                  x2={pxX}
                  y2={pxYPred}
                  stroke="#ef4444"
                  strokeWidth="2"
                />
              </g>
            );
          })}

          {/* Regresní pøímka */}
          <line
            x1={scaleX(xMin)}
            y1={scaleY(slope * xMin + intercept)}
            x2={scaleX(xMax)}
            y2={scaleY(slope * xMax + intercept)}
            stroke="#2563eb"
            strokeWidth="2.5"
          />

          {/* Datové body */}
          {DEFAULT_POINTS.map((pt, idx) => (
            <circle
              key={`point-${idx}`}
              cx={scaleX(pt.x)}
              cy={scaleY(pt.y)}
              r="4.5"
              fill="#0f172a"
            />
          ))}
        </svg>
      </div>

      {/* Ovládací prvky */}
      <div className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <label className="font-medium text-slate-700">Smìrnice (Slope, $\beta_1$)</label>
              <span className="font-mono text-slate-600">{slope.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={slope}
              onChange={(e) => setSlope(parseFloat(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer"
            />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <label className="font-medium text-slate-700">Posun (Intercept, $\beta_0$)</label>
              <span className="font-mono text-slate-600">{intercept.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-1.0"
              max="5.0"
              step="0.1"
              value={intercept}
              onChange={(e) => setIntercept(parseFloat(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-2">
          <label className="inline-flex items-center gap-2 text-sm text-slate-700 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showSquares}
              onChange={(e) => setShowSquares(e.target.checked)}
              className="rounded border-slate-300 text-blue-600 focus:ring-blue-500"
            />
            Zobrazit ètverce chyb ($e_i^2$)
          </label>

          <button
            onClick={handleSnapToOptimal}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md transition-colors shadow-sm"
          >
            Nastavit optimální OLS pøímku
          </button>
        </div>
      </div>
    </div>
  );
};