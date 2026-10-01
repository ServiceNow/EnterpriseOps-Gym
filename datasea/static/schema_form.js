// Generic JSON Schema -> HTML form renderer for MCP tool inputSchemas.
// buildForm(schema) returns {el, getValue()}; getValue() throws a user-facing Error on invalid input.
// Empty optional fields are omitted from the arguments entirely.

(function () {
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") e.className = v;
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(kid));
    return e;
  };

  const humanize = (name) =>
    String(name).replace(/([a-z])([A-Z])/g, "$1 $2").replace(/[_\-]+/g, " ").replace(/^./, (c) => c.toUpperCase());

  function resolve(schema, root) {
    let s = schema || {};
    let guard = 0;
    while (s.$ref && guard++ < 20) {
      const path = s.$ref.replace(/^#\//, "").split("/");
      let target = root;
      for (const p of path) target = target && target[p];
      const { $ref, ...rest } = s;
      s = { ...(target || {}), ...rest };
    }
    if (s.allOf) {
      const merged = { ...s };
      delete merged.allOf;
      for (const sub of s.allOf) {
        const r = resolve(sub, root);
        Object.assign(merged, r, { properties: { ...(merged.properties || {}), ...(r.properties || {}) } });
        merged.required = [...(merged.required || []), ...(r.required || [])];
      }
      s = merged;
    }
    return s;
  }

  // Normalise nullable unions: {anyOf:[X,{type:null}]} or {type:[X,"null"]} -> X
  function unwrapNullable(s, root) {
    const variants = s.anyOf || s.oneOf;
    if (variants) {
      const nonNull = variants.map((v) => resolve(v, root)).filter((v) => v.type !== "null");
      if (nonNull.length === 1) {
        const { anyOf, oneOf, ...rest } = s;
        return { ...nonNull[0], ...rest, title: s.title || nonNull[0].title, description: s.description || nonNull[0].description };
      }
      return { ...s, _variants: nonNull };
    }
    if (Array.isArray(s.type)) {
      const t = s.type.filter((x) => x !== "null");
      return { ...s, type: t.length === 1 ? t[0] : t };
    }
    return s;
  }

  const TIME_NAME = /(time|timestamp|date|_at$|At$|expir|until|since|start|end)/i;
  const LONG_TEXT = /(body|content|message|description|text|note|comment|signature|html|query|reason|summary|details)/i;

  function epochUnit(name, schema) {
    const d = `${name} ${schema.description || ""}`.toLowerCase();
    if (/millisec|\bms\b|_ms\b|epoch ms/.test(d)) return "ms";
    if (/\bseconds\b|unix time|unix timestamp/.test(d)) return "s";
    return "ms";
  }

  function field(name, rawSchema, root, required) {
    const s = unwrapNullable(resolve(rawSchema, root), root);
    const label = s.title && s.title !== humanize(name) ? s.title : humanize(name);
    const help = s.description ? h("div", { class: "help" }, s.description) : null;
    const reqMark = required ? h("span", { class: "req", title: "required" }, " *") : null;

    if (s._variants) {
      // Choice between several non-null shapes.
      const sel = h("select", { class: "variant" },
        h("option", { value: "" }, required ? "— choose a type —" : "— not set —"),
        s._variants.map((v, i) => h("option", { value: i }, v.title || v.type || `Option ${i + 1}`)));
      const slot = h("div", { class: "variant-slot" });
      let current = null;
      sel.addEventListener("change", () => {
        slot.innerHTML = "";
        current = sel.value === "" ? null : field(name, s._variants[+sel.value], root, true);
        if (current) slot.append(current.inner || current.el);
      });
      const el = h("div", { class: "field" }, h("label", {}, label, reqMark), help, sel, slot);
      return { el, getValue: () => {
        if (!current) { if (required) throw new Error(`${label}: choose an option`); return undefined; }
        return current.getValue();
      } };
    }

    const type = s.type || (s.properties ? "object" : s.items ? "array" : s.enum ? "string" : "string");

    if (s.enum) {
      const sel = h("select", {},
        h("option", { value: "" }, required ? "— choose —" : "— not set —"),
        s.enum.map((v, i) => h("option", { value: i }, String(v))));
      if (s.default !== undefined && required) sel.value = String(s.enum.indexOf(s.default));
      return wrap(label, reqMark, help, sel, () => {
        if (sel.value === "") { if (required) throw new Error(`${label} is required`); return undefined; }
        return s.enum[+sel.value];
      });
    }

    if (type === "boolean") {
      const sel = h("select", {},
        h("option", { value: "" }, required ? "— choose —" : "— not set —"),
        h("option", { value: "true" }, "Yes"), h("option", { value: "false" }, "No"));
      if (s.default !== undefined && required) sel.value = String(s.default);
      return wrap(label, reqMark, help, sel, () => {
        if (sel.value === "") { if (required) throw new Error(`${label} is required`); return undefined; }
        return sel.value === "true";
      });
    }

    if (type === "integer" || type === "number") {
      const inp = h("input", { type: "number", step: type === "integer" ? "1" : "any",
        placeholder: s.default !== undefined ? `default: ${s.default}` : "" , min: s.minimum, max: s.maximum });
      const extras = [];
      if (type === "integer" && TIME_NAME.test(name)) {
        // Helper so workers never compute epoch numbers by hand. Interpreted as UTC.
        const unit = epochUnit(name, s);
        const picker = h("input", { type: "datetime-local", step: "1" });
        const unitSel = h("select", {}, h("option", { value: "ms" }, "milliseconds"), h("option", { value: "s" }, "seconds"));
        unitSel.value = unit;
        const apply = () => {
          if (!picker.value) return;
          const ms = Date.parse(picker.value + "Z");
          inp.value = unitSel.value === "ms" ? ms : Math.floor(ms / 1000);
        };
        picker.addEventListener("change", apply);
        unitSel.addEventListener("change", apply);
        extras.push(h("div", { class: "time-helper" }, "Pick a date/time (UTC): ", picker, " as ", unitSel));
      }
      return wrap(label, reqMark, help, h("div", {}, inp, extras), () => {
        if (inp.value === "") { if (required) throw new Error(`${label} is required`); return undefined; }
        const n = Number(inp.value);
        if (type === "integer" && !Number.isInteger(n)) throw new Error(`${label} must be a whole number`);
        return n;
      });
    }

    if (type === "array") {
      const itemSchema = unwrapNullable(resolve(s.items || { type: "string" }, root), root);
      const list = h("div", { class: "array-items" });
      const rows = [];
      const add = () => {
        const f = field(`${name} item`, itemSchema, root, true);
        const row = h("div", { class: "array-row" }, f.inner || f.el);
        const rm = h("button", { type: "button", class: "small danger", onclick: () => {
          rows.splice(rows.indexOf(entry), 1); row.remove(); } }, "Remove");
        row.append(rm);
        const entry = { row, f };
        rows.push(entry);
        list.append(row);
      };
      const addBtn = h("button", { type: "button", class: "small", onclick: add }, "+ Add item");
      if (required && (s.minItems || 0) > 0) for (let i = 0; i < s.minItems; i++) add();
      return wrap(label, reqMark, help, h("div", {}, list, addBtn), () => {
        const vals = rows.map((r) => r.f.getValue()).filter((v) => v !== undefined);
        if (!vals.length) { if (required) throw new Error(`${label}: add at least one item`); return undefined; }
        return vals;
      });
    }

    if (type === "object") {
      if (s.properties && Object.keys(s.properties).length) {
        const sub = objectFields(s, root);
        const box = h("fieldset", { class: "nested" }, h("legend", {}, label, reqMark), help, sub.el);
        return { el: box, inner: box, getValue: () => {
          const v = sub.getValue();
          if (!Object.keys(v).length) { if (required) throw new Error(`${label} is required`); return undefined; }
          return v;
        } };
      }
      // Free-form map: key/value rows (values typed per additionalProperties when given).
      const valSchema = typeof s.additionalProperties === "object" ? s.additionalProperties : { type: "string" };
      const list = h("div", { class: "array-items" });
      const rows = [];
      const add = () => {
        const k = h("input", { type: "text", placeholder: "name" });
        const vf = field("value", valSchema, root, true);
        const row = h("div", { class: "array-row kv" }, k, vf.inner || vf.el);
        const entry = { k, vf };
        row.append(h("button", { type: "button", class: "small danger", onclick: () => { rows.splice(rows.indexOf(entry), 1); row.remove(); } }, "Remove"));
        rows.push(entry);
        list.append(row);
      };
      return wrap(label, reqMark, help, h("div", {}, list, h("button", { type: "button", class: "small", onclick: add }, "+ Add entry")), () => {
        const out = {};
        for (const r of rows) if (r.k.value.trim()) out[r.k.value.trim()] = r.vf.getValue();
        if (!Object.keys(out).length) { if (required) throw new Error(`${label} is required`); return undefined; }
        return out;
      });
    }

    // string
    let inp;
    const extras = [];
    if (s.format === "date") inp = h("input", { type: "date" });
    else if (s.format === "email") inp = h("input", { type: "email" });
    else if (LONG_TEXT.test(name) || (s.maxLength || 0) > 200) inp = h("textarea", { rows: 4 });
    else inp = h("input", { type: "text" });
    if (s.default !== undefined) inp.placeholder = `default: ${s.default}`;
    if (s.format === "date-time" || (TIME_NAME.test(name) && /iso ?8601|timestamp|date|time/i.test(s.description || ""))) {
      // Free text stays editable; the picker just fills it in the chosen format (UTC).
      const picker = h("input", { type: "datetime-local", step: "1" });
      const fmt = h("select", {},
        h("option", { value: "iso" }, "ISO 8601 (2025-01-31T09:00:00Z)"),
        h("option", { value: "ms" }, "epoch milliseconds"),
        h("option", { value: "s" }, "epoch seconds"));
      const apply = () => {
        if (!picker.value) return;
        const ms = Date.parse(picker.value + "Z");
        inp.value = fmt.value === "iso" ? new Date(ms).toISOString().replace(".000Z", "Z")
          : fmt.value === "ms" ? String(ms) : String(Math.floor(ms / 1000));
      };
      picker.addEventListener("change", apply);
      fmt.addEventListener("change", apply);
      extras.push(h("div", { class: "time-helper" }, "Pick a date/time (UTC): ", picker, " as ", fmt));
    }
    const pattern = s.pattern ? new RegExp(s.pattern) : null;
    return wrap(label, reqMark, help, h("div", {}, inp, extras), () => {
      const v = inp.value;
      if (v === "") { if (required) throw new Error(`${label} is required`); return undefined; }
      if (pattern && !pattern.test(v)) throw new Error(`${label}: "${v}" is not in the expected format`);
      if (s.minLength && v.length < s.minLength) throw new Error(`${label} is too short`);
      if (s.maxLength && v.length > s.maxLength) throw new Error(`${label} is too long (max ${s.maxLength})`);
      return v;
    });
  }

  function wrap(label, reqMark, help, input, getValue) {
    const el = h("div", { class: "field" }, h("label", {}, label, reqMark), help, input);
    return { el, getValue };
  }

  function objectFields(schema, root) {
    const req = new Set(schema.required || []);
    const props = Object.entries(schema.properties || {});
    props.sort((a, b) => (req.has(b[0]) - req.has(a[0])));
    const fields = props.map(([k, v]) => [k, field(k, v, root, req.has(k))]);
    const el = h("div", { class: "fields" }, fields.map(([, f]) => f.el));
    if (!fields.length) el.append(h("div", { class: "help" }, "This tool takes no inputs."));
    return {
      el,
      getValue: () => {
        const out = {};
        for (const [k, f] of fields) {
          const v = f.getValue();
          if (v !== undefined) out[k] = v;
        }
        return out;
      },
    };
  }

  window.SchemaForm = {
    buildForm(schema) {
      const root = schema || {};
      return objectFields(resolve(root, root), root);
    },
    humanize,
    h,
  };
})();
