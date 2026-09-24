// Nitro's prerender step spawns `<preview command> --port N --host localhost`,
// but `wrangler pages dev` takes --ip, not --host. Translate and forward.
import { spawn } from 'node:child_process';

const args = process.argv.slice(2);
const out = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--host') {
    out.push('--ip', args[++i] === 'localhost' ? '127.0.0.1' : args[i]);
  } else {
    out.push(args[i]);
  }
}

const child = spawn(
  'npx',
  [
    'wrangler',
    'pages',
    'dev',
    'dist',
    '--compatibility-flags',
    'nodejs_compat',
    '--compatibility-date',
    '2025-09-01',
    ...out,
  ],
  { stdio: 'inherit' },
);
child.on('exit', (code) => process.exit(code ?? 1));
for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => child.kill(sig));
}
