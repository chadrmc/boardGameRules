const API_BASE = "http://localhost:8000";

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Element {
  id: string;
  rulebook_id: string;
  source_type: string;
  page_number: number;
  display_mode: "image" | "text";
  page_image_path: string;
  type: string;
  label: string;
  description: string;
  bbox: BBox;
}

export interface SearchResult {
  element: Element;
  score: number;
  errata: Element[];
  faq: Element[];
}

export interface Rulebook {
  id: string;
  name: string;
}

export interface IngestRulebookResult {
  rulebook_id: string;
  pages_processed: number;
  elements_found: number;
}

export async function getPageElements(rulebookId: string, pageNumber: number): Promise<Element[]> {
  const params = new URLSearchParams({ rulebook_id: rulebookId, page_number: String(pageNumber) });
  const res = await fetch(`${API_BASE}/elements?${params}`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.elements;
}

export async function listRulebooks(): Promise<Rulebook[]> {
  const res = await fetch(`${API_BASE}/rulebooks`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  return data.rulebooks;
}

export async function ask(
  query: string,
  rulebookId: string,
  onToken: (text: string) => void,
  onResults: (results: SearchResult[]) => void,
  n = 5,
): Promise<void> {
  const params = new URLSearchParams({ q: query, rulebook_id: rulebookId, n: String(n) });
  const res = await fetch(`${API_BASE}/ask?${params}`);
  if (!res.ok) throw new Error(await res.text());

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = JSON.parse(line.slice(6));
      if (data.type === "results") onResults(data.results);
      else if (data.type === "token") onToken(data.text);
    }
  }
}

export async function ingestRulebook(
  rulebookId: string,
  file: File,
  sourceType = "core",
  gameName?: string,
  onProgress?: (page: number, total: number, elementsOnPage: number) => void,
): Promise<IngestRulebookResult> {
  const params = new URLSearchParams({ source_type: sourceType });
  if (gameName) params.set("game_name", gameName);
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/rulebooks/${rulebookId}?${params}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(await res.text());

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: IngestRulebookResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = JSON.parse(line.slice(6));
      if (data.done) {
        result = {
          rulebook_id: data.rulebook_id,
          pages_processed: data.pages_processed,
          elements_found: data.elements_found,
        };
      } else {
        onProgress?.(data.page, data.total, data.elements);
      }
    }
  }

  if (!result) throw new Error("No result received from server");
  return result;
}
