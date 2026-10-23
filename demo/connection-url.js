// @ts-check

/**
 * Private HF Space embeds receive a short-lived __sign grant on the page URL.
 * WebSockets cannot attach an Authorization header, and iframe cookies may be
 * unavailable. Carry that page grant only to this Space's own secure socket.
 * Never forward a page credential to a direct, external inference endpoint.
 * @param {string} socketUrl
 * @param {string | undefined} pageUrl
 */
export function withSpaceAccess(socketUrl, pageUrl = globalThis.location?.href) {
  if (!pageUrl) return socketUrl;
  const page = new URL(pageUrl);
  const socket = new URL(socketUrl, page);
  if (page.protocol !== "https:" || socket.protocol !== "wss:" || socket.host !== page.host) {
    return socketUrl;
  }
  const grant = page.searchParams.get("__sign");
  if (grant && !socket.searchParams.has("__sign")) socket.searchParams.set("__sign", grant);
  return socket.href;
}
