"""X-ray inference çıktılarını DocumentAnalysis formatına map eden saf fonksiyonlar.

`predict_chest` / `predict_bone` dict çıktılarını alıp:
  - severity
  - summary
  - findings  (her biri: label, probability, description, recommendation)
  - recommendations
  - type_specific
alanlarını üretir.

LLM çağrısı YOK — bu modül deterministik ve saf. Kişiselleştirilmiş AI özeti
ayrı bir aşamada (core.llm.summarize_xray_findings) eklenir.
"""

from __future__ import annotations

from typing import Final, TypedDict

# ─── Tipler ──────────────────────────────────────────────────────────────────

class FindingInfo(TypedDict):
    description: str
    recommendation: str


# ─── CHEST: bulgu bilgileri ──────────────────────────────────────────────────

CHEST_FINDING_INFO: Final[dict[str, FindingInfo]] = {
    "Cardiomegaly": {
        "description": (
            "Kalbin normalden büyük görünmesi. Uzun süreli hipertansiyon, "
            "kalp kapak hastalıkları veya kalp yetmezliği gibi durumların "
            "işareti olabilir. Tek başına tanı koydurmaz, ileri tetkik gerekir."
        ),
        "recommendation": "Kardiyoloji konsültasyonu önerilir; EKG ve ekokardiyografi istenmesi beklenir.",
    },
    "Effusion": {
        "description": (
            "Akciğer ile göğüs duvarı arasındaki boşlukta sıvı birikimi "
            "(plevral efüzyon). Enfeksiyon, kalp yetmezliği, böbrek hastalığı "
            "veya tümöral süreçlerle ilişkili olabilir."
        ),
        "recommendation": "Plevral efüzyon için göğüs hastalıkları uzmanına başvurun; nedene yönelik tetkik gerekir.",
    },
    "Pneumonia": {
        "description": (
            "Akciğer dokusunda enfeksiyon bulgusu (zatürre). Bakteriyel, viral "
            "veya mantar kaynaklı olabilir; ateş, öksürük ve nefes darlığı ile "
            "birlikte gidebilir."
        ),
        "recommendation": "Enfeksiyon hastalıkları veya göğüs hastalıkları uzmanına başvurun; antibiyotik tedavisi gerekebilir.",
    },
    "Pneumothorax": {
        "description": (
            "Plevral boşlukta hava birikmesi sonucu akciğerin kısmen veya "
            "tamamen sönmesi. Ani göğüs ağrısı ve nefes darlığı yapar; "
            "boyutuna göre acil müdahale gerektirebilir."
        ),
        "recommendation": "ACİL: Vakit kaybetmeden acil servise başvurun.",
    },
    "Atelectasis": {
        "description": (
            "Akciğerin bir bölümünün havasız kalması, yani sönmesi. "
            "Genellikle uzun süre yatmaya, mukus tıkacına veya ameliyat "
            "sonrası döneme bağlı gelişir; ciddi bir tıkanıklığa da bağlı olabilir."
        ),
        "recommendation": "Göğüs hastalıkları uzmanı veya solunum fizyoterapisti değerlendirmesi önerilir.",
    },
    "Consolidation": {
        "description": (
            "Akciğer dokusunun normalde hava içeren bölgelerinin sıvı, irin "
            "veya hücresel materyalle dolması. En sık zatürre ile birlikte "
            "görülür; tümör veya kanama da neden olabilir."
        ),
        "recommendation": "Göğüs hastalıkları uzmanına başvurun; klinik değerlendirme ve ileri görüntüleme gerekebilir.",
    },
    "Edema": {
        "description": (
            "Akciğer dokusunda sıvı birikimi (akciğer ödemi). En sık kalp "
            "yetmezliğine bağlı görülür; böbrek yetmezliği veya yüksek irtifa "
            "gibi nedenleri de olabilir. Nefes darlığı temel belirtisidir."
        ),
        "recommendation": "Kardiyoloji veya nefroloji değerlendirmesi gereklidir; altta yatan neden araştırılmalıdır.",
    },
    "Emphysema": {
        "description": (
            "Akciğerdeki küçük hava keseciklerinin kalıcı olarak genişlemesi "
            "ve duvarlarının hasar görmesi. KOAH'ın bir bileşenidir, en sık "
            "uzun süreli sigara kullanımına bağlıdır; geri dönüşümsüz nefes "
            "darlığı yapar."
        ),
        "recommendation": "Göğüs hastalıkları uzmanına başvurun; solunum fonksiyon testi (SFT) önerilir.",
    },
    "Fibrosis": {
        "description": (
            "Akciğer dokusunda kalıcı yara/sertleşme oluşması (pulmoner "
            "fibrozis). Geçirilmiş enfeksiyon, mesleki maruziyet, otoimmün "
            "hastalık veya idiyopatik (nedeni bilinmeyen) olabilir; ilerleyici "
            "nefes darlığı yapar."
        ),
        "recommendation": "Göğüs hastalıkları uzmanına başvurun; yüksek çözünürlüklü BT (HRCT) gerekebilir.",
    },
    "Hernia": {
        "description": (
            "Karın içi organlarının (en sık mide) diyaframdaki açıklıktan "
            "göğüs boşluğuna doğru yer değiştirmesi (diyafragma fıtığı). "
            "Reflü, göğüs ağrısı ve yutma güçlüğüne yol açabilir."
        ),
        "recommendation": "Genel cerrahi değerlendirmesi önerilir; gastroenteroloji konsültasyonu da yararlı olabilir.",
    },
    "Infiltration": {
        "description": (
            "Akciğer dokusunda normalde olmaması gereken sıvı, hücre veya "
            "madde birikimi. Enfeksiyon, inflamasyon, alerjik reaksiyon veya "
            "tümöral süreçlerin işareti olabilir."
        ),
        "recommendation": "Enfeksiyon veya inflamasyon için klinik değerlendirme gereklidir; göğüs hastalıkları uzmanı görmelidir.",
    },
    "Mass": {
        "description": (
            "Akciğerde 3 cm'den büyük yer kaplayan oluşum. İyi huylu "
            "(benign) olabileceği gibi habis (malign/kanseröz) de olabilir; "
            "her durumda ileri tetkikle ayırt edilmesi gerekir."
        ),
        "recommendation": "ACİL: Onkoloji veya göğüs cerrahisi konsültasyonu gereklidir; BT ve biyopsi planlanmalıdır.",
    },
    "Nodule": {
        "description": (
            "Akciğerde 3 cm'den küçük yuvarlak/oval lezyon. Çoğunluğu iyi "
            "huyludur (eski enfeksiyon izleri, granülom) ancak erken evre "
            "akciğer kanseri de olabilir; mutlaka takip gerekir."
        ),
        "recommendation": "Takip görüntülemesi ve göğüs hastalıkları değerlendirmesi önerilir; BT ile karakterizasyon gerekebilir.",
    },
    "Pleural_Thickening": {
        "description": (
            "Akciğer zarının (plevra) kalınlaşması. Geçirilmiş enfeksiyon, "
            "asbest gibi mesleki maruziyetler veya kronik inflamasyona bağlı "
            "gelişebilir; ileri evrede solunum kısıtlılığı yapabilir."
        ),
        "recommendation": "Göğüs hastalıkları uzmanına başvurun; mesleki maruziyet öyküsü sorgulanmalıdır.",
    },
}

_GENERIC_CHEST_INFO: Final[FindingInfo] = {
    "description": "Akciğer grafisinde dikkat çeken bir bulgu saptandı.",
    "recommendation": "Uzman hekime başvurmanız önerilir.",
}

_NORMAL_PROB_THRESHOLD: Final[float] = 0.7
_CRITICAL_FINDING_THRESHOLD: Final[float] = 0.75


# ─── CHEST: helpers ──────────────────────────────────────────────────────────

def _chest_severity(top_findings: list[dict], normal_probability: float) -> str:
    if not top_findings or normal_probability > _NORMAL_PROB_THRESHOLD:
        return "normal"
    if any(f.get("probability", 0.0) > _CRITICAL_FINDING_THRESHOLD for f in top_findings):
        return "critical"
    return "warning"


def _chest_summary(top_findings: list[dict]) -> str:
    if not top_findings:
        return "Röntgen analizi tamamlandı. Belirgin bir patolojik bulgu saptanmadı."

    parts = [
        f"{f.get('label', '')} (%{round(f.get('probability', 0.0) * 100)})"
        for f in top_findings
    ]
    return "Röntgen analizinde şu bulgular tespit edildi: " + ", ".join(parts) + "."


_LOW_CONFIDENCE_NOTE: Final[str] = (
    " Model bu bulgu için emin değil (kalibrasyon verisine göre ayrım kabiliyeti "
    "sınırlı); doktor değerlendirmesi şart."
)

_CHEST_OOD_WARNING_TEXT: Final[str] = (
    "Yüklenen görüntü modelin eğitildiği akciğer röntgeni dağılımının dışında "
    "görünüyor. Bu, BT/MR/USG gibi farklı bir modalitenin veya hatalı bir "
    "görüntünün yüklenmiş olabileceği anlamına gelir. Sonuçlar güvenilir "
    "olmayabilir — lütfen düz akciğer röntgeni (PA/AP) yüklediğinizden emin olun."
)


def _enrich_chest_findings(top_findings: list[dict]) -> list[dict]:
    enriched: list[dict] = []
    for f in top_findings:
        label = f.get("label", "")
        info = CHEST_FINDING_INFO.get(label, _GENERIC_CHEST_INFO)
        is_low_conf = bool(f.get("low_confidence", False))

        description = info["description"]
        recommendation = info["recommendation"]
        if is_low_conf:
            description = description + _LOW_CONFIDENCE_NOTE

        item: dict = {
            "label":          label,
            "probability":    float(f.get("probability", 0.0)),
            "description":    description,
            "recommendation": recommendation,
        }
        if is_low_conf:
            item["low_confidence"] = True
        heatmap = f.get("heatmap_base64")
        if heatmap:
            item["heatmap_base64"] = heatmap
        enriched.append(item)
    return enriched


def _chest_recommendations(enriched_findings: list[dict]) -> list[str]:
    recs: list[str] = []
    seen: set[str] = set()
    for f in enriched_findings:
        rec = f.get("recommendation", "")
        if rec and rec not in seen:
            recs.append(rec)
            seen.add(rec)
    return recs


def map_chest_to_analysis(chest_result: dict) -> dict:
    """`predict_chest` çıktısını DocumentAnalysis formatına çevirir."""
    top_findings: list[dict] = list(chest_result.get("top_findings") or [])
    normal_probability = float(chest_result.get("normal_probability", 0.0))
    is_ood = bool(chest_result.get("ood_is_out_of_distribution", False))

    severity = _chest_severity(top_findings, normal_probability)
    # OOD ise critical severity'yi indir — model güvensiz
    if is_ood and severity == "critical":
        severity = "warning"

    summary = _chest_summary(top_findings)
    if is_ood:
        summary = "⚠ Kapsam dışı görüntü uyarısı. " + summary

    findings = _enrich_chest_findings(top_findings)
    recommendations = _chest_recommendations(findings)
    if is_ood:
        recommendations.insert(0, _CHEST_OOD_WARNING_TEXT)

    type_specific = {**chest_result, "xray_type": "chest"}

    return {
        "severity":        severity,
        "summary":         summary,
        "findings":        findings,
        "recommendations": recommendations,
        "type_specific":   type_specific,
    }


# ─── BONE ────────────────────────────────────────────────────────────────────

_BONE_NORMAL_DESCRIPTION: Final[str] = (
    "Kemik röntgeninde belirgin bir kırık, çıkık veya yapısal anormallik "
    "saptanmadı. Görüntü modelin değerlendirmesine göre normal sınırlar içindedir."
)

_BONE_UNRELIABLE_DESCRIPTION: Final[str] = (
    "Bu görüntü için model çıktısı güvenilir kabul edilmemelidir. Görüntü dağılım dışı "
    "veya düşük güvenli görünüyor; bu durumda 'normal' sonucu klinik olarak rahatlatıcı "
    "bir sonuç sayılmaz. Radyoloji/ortopedi uzmanı tarafından doğrudan değerlendirilmelidir."
)

_BONE_ABNORMAL_DESCRIPTION_HIGH: Final[str] = (
    "Kemik röntgeninde yüksek olasılıkla anormallik (kırık, çıkık veya "
    "yapısal bozukluk) tespit edildi. Modelin güveni yüksek; klinik tablo "
    "ile birlikte mutlaka acil değerlendirme yapılmalıdır."
)

_BONE_ABNORMAL_DESCRIPTION_MIDLOW: Final[str] = (
    "Kemik röntgeninde olası bir anormallik şüphesi tespit edildi. Modelin "
    "güven düzeyi yüksek değil; bu nedenle bulgunun klinik bulgular ve "
    "uzman görüşü ile birlikte değerlendirilmesi gerekir."
)


def _bone_severity(result: str, confidence: str) -> str:
    if confidence == "low":
        return "warning"
    if result == "normal":
        return "normal"
    if result == "abnormal" and confidence == "high":
        return "critical"
    return "warning"


def _bone_summary(result: str, confidence: str) -> str:
    if confidence == "low":
        return "Model bu kemik röntgenini düşük güvenle değerlendirdi; sonuç güvenilir değildir."
    if result == "normal":
        return "Kemik röntgeni normal sınırlar içinde değerlendirildi."
    return f"Kemik röntgeninde anormallik tespit edildi (güven: {confidence})."


def _bone_description(result: str, confidence: str) -> str:
    if confidence == "low":
        return _BONE_UNRELIABLE_DESCRIPTION
    if result == "normal":
        return _BONE_NORMAL_DESCRIPTION
    if confidence == "high":
        return _BONE_ABNORMAL_DESCRIPTION_HIGH
    return _BONE_ABNORMAL_DESCRIPTION_MIDLOW


def _bone_recommendations(result: str, confidence: str) -> list[str]:
    if confidence == "low":
        return [
            "Radyoloji raporu ve ortopedi değerlendirmesi olmadan bu sonucu normal kabul etmeyin.",
            "Ağrı, şişlik, şekil bozukluğu, travma veya hareket kısıtlılığı varsa acil/ortopedi başvurusu yapın.",
        ]
    if result == "normal":
        return ["Düzenli kontrol muayenelerini ihmal etmeyin."]
    if confidence == "high":
        return [
            "ACİL: Ortopedi veya acil servise başvurun.",
            "Etkilenen bölgeyi hareket ettirmeyin.",
        ]
    return [
        "Ortopedi uzmanına başvurmanız önerilir.",
        "Ağır yük taşımaktan kaçının.",
    ]


_OOD_WARNING_TEXT: Final[str] = (
    "Yüklenen görüntü modelin eğitildiği veri dağılımının dışında görünüyor. "
    "Bone modeli yalnızca üst ekstremite röntgenleri (el, bilek, dirsek, ön kol, "
    "üst kol, omuz, parmak) için güvenilir sonuç verir. Tahmin düşük güvenle "
    "değerlendirilmelidir — sonucu klinik bulgularla birlikte yorumlayın."
)


def map_bone_to_analysis(bone_result: dict) -> dict:
    """`predict_bone` çıktısını DocumentAnalysis formatına çevirir."""
    result = str(bone_result.get("result", "normal"))
    confidence = str(bone_result.get("confidence", "low"))
    abnormal_probability = float(bone_result.get("abnormal_probability", 0.0))
    is_ood = bool(bone_result.get("ood_is_out_of_distribution", False))
    region_label = str(bone_result.get("bone_region_label", "") or "")

    severity = _bone_severity(result, confidence)
    if is_ood:
        severity = "warning"

    summary = _bone_summary(result, confidence)
    if region_label:
        summary = f"[{region_label}] {summary}"
    if is_ood:
        summary = "⚠ Kapsam dışı görüntü uyarısı. " + summary

    description = _bone_description(result, confidence)
    recommendations = list(_bone_recommendations(result, confidence))
    if is_ood:
        recommendations.insert(0, _OOD_WARNING_TEXT)

    finding: dict = {
        "label":          "Güvenilir Değerlendirme Yok" if is_ood or confidence == "low" else ("Anormallik" if result == "abnormal" else "Normal Bulgu"),
        "probability":    abnormal_probability,
        "description":    description,
        "recommendation": recommendations[0] if recommendations else "",
    }
    heatmap = bone_result.get("heatmap_base64")
    if heatmap:
        finding["heatmap_base64"] = heatmap
    findings = [finding]

    type_specific = {**bone_result, "xray_type": "bone"}

    return {
        "severity":        severity,
        "summary":         summary,
        "findings":        findings,
        "recommendations": recommendations,
        "type_specific":   type_specific,
    }
