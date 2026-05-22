import {Composition} from 'remotion';
import {KernelForgeDemo} from './Video';

export const RemotionRoot: React.FC = () => (
  <Composition
    id="KernelForgeDemo"
    component={KernelForgeDemo}
    durationInFrames={30 * 135}
    fps={30}
    width={1920}
    height={1080}
    defaultProps={{}}
  />
);
