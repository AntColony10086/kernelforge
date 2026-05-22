import {AbsoluteFill, Sequence, useVideoConfig, Audio, staticFile, useCurrentFrame, interpolate} from 'remotion';
import React from 'react';

const BG = '#0b0b0c';
const FG = '#eaeaf0';
const ACCENT_GREEN = '#52d273';
const ACCENT_RED = '#f86b6b';
const ACCENT_BLUE = '#6cb8ff';
const MONO = '"JetBrains Mono", "SF Mono", Menlo, monospace';

export const KernelForgeDemo: React.FC = () => {
  const {fps} = useVideoConfig();
  return (
    <AbsoluteFill style={{backgroundColor: BG, color: FG, fontFamily: MONO}}>
      <Sequence from={0} durationInFrames={fps * 25}>
        <Opener />
      </Sequence>
      <Sequence from={fps * 25} durationInFrames={fps * 30}>
        <EscalationBeat />
      </Sequence>
      <Sequence from={fps * 55} durationInFrames={fps * 50}>
        <MoneyShot />
      </Sequence>
      <Sequence from={fps * 105} durationInFrames={fps * 15}>
        <ScaleShot />
      </Sequence>
      <Sequence from={fps * 120} durationInFrames={fps * 15}>
        <Scorecard />
      </Sequence>
      <TryAudio />
    </AbsoluteFill>
  );
};

const TryAudio: React.FC = () => {
  // Audio is optional; if voiceover.wav is not present, the video still renders silent.
  try {
    return <Audio src={staticFile('voiceover.wav')} />;
  } catch {
    return null;
  }
};

const Title: React.FC<{children: React.ReactNode; sub?: string}> = ({children, sub}) => (
  <div style={{display: 'flex', flexDirection: 'column', gap: 16}}>
    <div style={{fontSize: 56, fontWeight: 700, letterSpacing: -1}}>{children}</div>
    {sub && <div style={{fontSize: 24, opacity: 0.7}}>{sub}</div>}
  </div>
);

const Opener: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 15], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', padding: 100, textAlign: 'left', opacity}}>
      <div style={{maxWidth: 1400}}>
        <Title sub="DevNetwork [AI+ML] Hackathon 2026 — TrueFoundry Resilient Agents track">KernelForge</Title>
        <div style={{marginTop: 60, fontSize: 36, lineHeight: 1.4}}>
          <span style={{color: ACCENT_RED}}>The worst kernel failure is not a crash.</span><br/>
          It's an LLM-written kernel that compiles, passes a smoke test,<br/>
          and silently produces wrong outputs on the next shape.
        </div>
      </div>
    </AbsoluteFill>
  );
};

const EscalationBeat: React.FC = () => (
  <AbsoluteFill style={{padding: 80, flexDirection: 'column', gap: 32}}>
    <Title sub="Beat 2 — Cost-aware LLM escalation via TrueFoundry AI Gateway">LLM brownout / failure</Title>
    <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginTop: 24}}>
      <div style={{background: '#1a1010', padding: 24, borderRadius: 12, border: `1px solid ${ACCENT_RED}66`}}>
        <div style={{color: ACCENT_RED, fontSize: 20, marginBottom: 12}}>Naive baseline</div>
        <pre style={{fontSize: 18, lineHeight: 1.6}}>{`POST deepseek-v4-flash
HTTP 429 (rate limit)
agent halted.`}</pre>
      </div>
      <div style={{background: '#101a14', padding: 24, borderRadius: 12, border: `1px solid ${ACCENT_GREEN}66`}}>
        <div style={{color: ACCENT_GREEN, fontSize: 20, marginBottom: 12}}>KernelForge + TrueFoundry</div>
        <pre style={{fontSize: 18, lineHeight: 1.6}}>{`POST deepseek-v4-flash
HTTP 429 -> routing applied
x-tfy-routing: from=v4-flash
              to=v4-pro
              reason=quality-escalation
agent continues.`}</pre>
      </div>
    </div>
    <div style={{marginTop: 24, fontSize: 22, opacity: 0.85}}>
      Routing rule in <code>routing_config.yaml</code>: <span style={{color: ACCENT_BLUE}}>fallback_status_codes</span> = [408, 429, 500, 502, 503, 504] &nbsp; <span style={{color: ACCENT_BLUE}}>fallback_candidate</span> = deepseek-v4-pro
    </div>
  </AbsoluteFill>
);

const MoneyShot: React.FC = () => (
  <AbsoluteFill style={{padding: 80, flexDirection: 'column', gap: 24}}>
    <Title sub="Beat 3 — same kernel, same DeepSeek output, two different verifiers">Money shot: hidden-holdout catches the silent bug</Title>
    <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32, marginTop: 16}}>
      <div style={{background: '#1a1010', padding: 24, borderRadius: 12, border: `1px solid ${ACCENT_RED}66`}}>
        <div style={{color: ACCENT_RED, fontSize: 22, marginBottom: 12}}>Naive (smoke test only)</div>
        <pre style={{fontSize: 16, lineHeight: 1.6}}>{`> generate rope kernel via v4-flash
> compile: OK
> smoke run on [1,8,64]: shape OK
RoPE kernel ready: 1.4x speedup ✓
[shipped to production]`}</pre>
      </div>
      <div style={{background: '#101a14', padding: 24, borderRadius: 12, border: `1px solid ${ACCENT_GREEN}66`}}>
        <div style={{color: ACCENT_GREEN, fontSize: 22, marginBottom: 12}}>KernelForge (hidden holdouts)</div>
        <pre style={{fontSize: 16, lineHeight: 1.6}}>{`> generate rope kernel via v4-flash
> compile: OK
> smoke run on [1,8,64]: shape OK
> holdout [2,32,128]: max_abs_diff 0.84
  ! suspected: split-half vs interleaved layout
> escalate to v4-pro, iter 2
> compile + holdouts: 5/5 pass
RoPE: verified correct after 2 iter`}</pre>
      </div>
    </div>
    <div style={{marginTop: 16, fontSize: 22, color: ACCENT_BLUE, textAlign: 'center'}}>
      TrueFoundry handles LLM provider resilience. KernelForge handles kernel correctness verification.
    </div>
  </AbsoluteFill>
);

const ScaleShot: React.FC = () => (
  <AbsoluteFill style={{padding: 80, flexDirection: 'column', gap: 24}}>
    <Title sub="Beat 4 — 3 ops, honest per-op perf disclosure">Scale across the benchmark suite</Title>
    <div style={{display: 'flex', flexDirection: 'column', gap: 16, marginTop: 24, fontSize: 26}}>
      <div>RoPE — verified after 2 iterations &nbsp;|&nbsp; <span style={{color: ACCENT_GREEN}}>1.2x</span> MLX eager &nbsp;|&nbsp; <span style={{color: ACCENT_RED}}>0.8x</span> mx.fast.rope (honest)</div>
      <div>RMSNorm — verified iter 1 &nbsp;|&nbsp; <span style={{color: ACCENT_GREEN}}>1.4x</span> MLX eager &nbsp;|&nbsp; <span style={{color: ACCENT_RED}}>0.7x</span> mx.fast.rms_norm</div>
      <div>SwiGLU — verified iter 1 &nbsp;|&nbsp; <span style={{color: ACCENT_GREEN}}>1.1x</span> MLX eager</div>
    </div>
    <div style={{marginTop: 24, fontSize: 20, opacity: 0.7}}>
      Where we lose to MLX built-ins, we say so. No PyTorch CPU comparisons.
    </div>
  </AbsoluteFill>
);

const Scorecard: React.FC = () => (
  <AbsoluteFill style={{padding: 80, justifyContent: 'center'}}>
    <div style={{maxWidth: 1400, margin: '0 auto'}}>
      <Title sub="Beat 5 — 4-row scorecard">Naive vs KernelForge</Title>
      <table style={{marginTop: 48, width: '100%', fontSize: 26, borderCollapse: 'collapse'}}>
        <thead>
          <tr>
            <th style={{textAlign: 'left', padding: 16, borderBottom: '2px solid #333'}}>Metric</th>
            <th style={{textAlign: 'left', padding: 16, borderBottom: '2px solid #333', color: ACCENT_RED}}>Naive</th>
            <th style={{textAlign: 'left', padding: 16, borderBottom: '2px solid #333', color: ACCENT_GREEN}}>KernelForge</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style={{padding: 16}}>Kernels claimed correct</td><td style={{padding: 16}}>3/3</td><td style={{padding: 16}}>3/3</td></tr>
          <tr><td style={{padding: 16}}>Hidden holdout pass rate</td><td style={{padding: 16, color: ACCENT_RED}}>33%</td><td style={{padding: 16, color: ACCENT_GREEN}}>100%</td></tr>
          <tr><td style={{padding: 16}}>Silent wrong-output rate</td><td style={{padding: 16, color: ACCENT_RED}}>67%</td><td style={{padding: 16, color: ACCENT_GREEN}}>0%</td></tr>
          <tr><td style={{padding: 16}}>LLM routing (TrueFoundry)</td><td style={{padding: 16}}>v4-flash only</td><td style={{padding: 16}}>v4-flash → v4-pro</td></tr>
        </tbody>
      </table>
      <div style={{marginTop: 60, fontSize: 28, opacity: 0.85}}>
        Correctness isn't a vibe. It's a holdout suite.
      </div>
      <div style={{marginTop: 16, fontSize: 22, color: ACCENT_BLUE}}>github.com/AntColony10086/kernelforge</div>
    </div>
  </AbsoluteFill>
);
