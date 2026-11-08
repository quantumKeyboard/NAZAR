import { useSyncExternalStore } from 'react';

/**
 * A shared one-second clock. Every component that shows elapsed or remaining time reads the
 * same value, and the timer only runs while somebody is subscribed.
 */
const listeners = new Set<() => void>();
let now = Date.now();
let timer: ReturnType<typeof setInterval> | undefined;

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (timer === undefined) {
    now = Date.now();
    timer = setInterval(() => {
      now = Date.now();
      listeners.forEach((l) => {
        l();
      });
    }, 1000);
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && timer !== undefined) {
      clearInterval(timer);
      timer = undefined;
    }
  };
}

const idle = () => () => undefined;
const snapshot = () => now;

/** The current time in ms, refreshed every second while `active`; frozen otherwise. */
export function useNow(active: boolean): number {
  return useSyncExternalStore(active ? subscribe : idle, snapshot);
}
