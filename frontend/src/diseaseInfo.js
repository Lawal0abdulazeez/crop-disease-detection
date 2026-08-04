/**
 * Lightweight treatment / info map for PlantVillage-style class names.
 * Keys are matched case-insensitively / partially against predicted class.
 */
const INFO = [
  {
    match: /healthy/i,
    title: "Healthy",
    summary: "No disease detected. Keep monitoring and maintain good practices.",
    treatment: [
      "Continue regular watering and balanced fertilization",
      "Ensure good air circulation around plants",
      "Inspect leaves weekly for early symptoms",
    ],
  },
  {
    match: /early.?blight/i,
    title: "Early Blight",
    summary: "Fungal disease causing brown spots with concentric rings.",
    treatment: [
      "Remove and destroy infected leaves",
      "Avoid overhead watering; water at soil level",
      "Apply copper-based or chlorothalonil fungicide as labeled",
      "Rotate crops; improve spacing for airflow",
    ],
  },
  {
    match: /late.?blight/i,
    title: "Late Blight",
    summary: "Aggressive fungal disease; can destroy crops quickly in humid weather.",
    treatment: [
      "Remove infected plants immediately",
      "Avoid working plants when wet",
      "Use approved late-blight fungicides promptly",
      "Do not compost infected material",
    ],
  },
  {
    match: /bacterial.?spot/i,
    title: "Bacterial Spot",
    summary: "Bacterial infection causing small dark lesions on leaves and fruit.",
    treatment: [
      "Remove infected foliage",
      "Avoid working wet plants",
      "Copper sprays may help reduce spread",
      "Use disease-free seed and resistant varieties when possible",
    ],
  },
  {
    match: /leaf.?mold/i,
    title: "Leaf Mold",
    summary: "Fungal issue common in humid greenhouses; pale spots then mold.",
    treatment: [
      "Increase ventilation and reduce humidity",
      "Space plants adequately",
      "Remove affected leaves",
      "Apply fungicide labeled for leaf mold if needed",
    ],
  },
  {
    match: /septoria/i,
    title: "Septoria Leaf Spot",
    summary: "Small circular spots with dark borders; spreads upward from lower leaves.",
    treatment: [
      "Remove lower infected leaves",
      "Mulch to reduce soil splash",
      "Avoid overhead irrigation",
      "Fungicides (e.g. chlorothalonil) as directed",
    ],
  },
  {
    match: /spider.?mite|two.?spotted/i,
    title: "Spider Mite Damage",
    summary: "Tiny pests causing stippling and fine webbing under leaves.",
    treatment: [
      "Spray undersides of leaves with water regularly",
      "Use insecticidal soap or neem oil",
      "Introduce predatory mites if available",
      "Avoid excess nitrogen fertilizer",
    ],
  },
  {
    match: /target.?spot/i,
    title: "Target Spot",
    summary: "Fungal lesions with concentric rings; thrives in warm, wet conditions.",
    treatment: [
      "Improve airflow and reduce leaf wetness",
      "Remove infected debris",
      "Rotate fungicide modes of action",
    ],
  },
  {
    match: /mosaic|virus/i,
    title: "Virus (e.g. Mosaic)",
    summary: "Viral infection; mottled leaves, stunting. No chemical cure.",
    treatment: [
      "Remove and destroy infected plants",
      "Control aphids and other insect vectors",
      "Sanitize tools between plants",
      "Plant resistant varieties next season",
    ],
  },
  {
    match: /yellow.?leaf.?curl/i,
    title: "Yellow Leaf Curl",
    summary: "Virus spread by whiteflies; yellowing and upward curling of leaves.",
    treatment: [
      "Control whiteflies (traps, approved insecticides)",
      "Remove severely infected plants",
      "Use reflective mulches / screens where practical",
      "Choose resistant cultivars",
    ],
  },
  {
    match: /scab/i,
    title: "Scab",
    summary: "Rough, corky lesions on fruit or leaves.",
    treatment: [
      "Remove infected plant parts",
      "Improve drainage and avoid prolonged leaf wetness",
      "Apply appropriate fungicides during susceptible periods",
    ],
  },
  {
    match: /rust/i,
    title: "Rust",
    summary: "Orange/brown pustules on undersides of leaves.",
    treatment: [
      "Remove infected leaves",
      "Avoid overhead watering",
      "Apply sulfur or other labeled fungicides",
      "Plant resistant varieties",
    ],
  },
  {
    match: /black.?rot/i,
    title: "Black Rot",
    summary: "V-shaped lesions from leaf margins; can affect fruit.",
    treatment: [
      "Remove infected leaves and fruit",
      "Rotate crops; avoid crucifer residues",
      "Use copper or other labeled products early",
    ],
  },
  {
    match: /powdery.?mildew/i,
    title: "Powdery Mildew",
    summary: "White powdery coating on leaves.",
    treatment: [
      "Improve air circulation",
      "Avoid excess nitrogen",
      "Sulfur or potassium bicarbonate sprays as labeled",
    ],
  },
];

const FALLBACK = {
  title: "Disease detected",
  summary:
    "Follow general integrated pest management. Confirm diagnosis with a local extension service when possible.",
  treatment: [
    "Isolate or remove heavily infected material",
    "Improve airflow and avoid prolonged leaf wetness",
    "Use labeled products only, following local regulations",
    "Monitor neighboring plants for spread",
  ],
};

export function getDiseaseInfo(className) {
  if (!className) return FALLBACK;
  const found = INFO.find((item) => item.match.test(className));
  if (!found) {
    return {
      ...FALLBACK,
      title: className.replace(/[_-]/g, " "),
    };
  }
  return {
    title: found.title,
    summary: found.summary,
    treatment: found.treatment,
  };
}
