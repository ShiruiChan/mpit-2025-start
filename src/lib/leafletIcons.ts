import L from "leaflet";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import iconRetinaUrl from "leaflet/dist/images/marker-icon-2x.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";

export function setupLeafletDefaultIcon() {
  const Default = L.Icon.Default as any;
  Default.mergeOptions({
    iconRetinaUrl,
    iconUrl,
    shadowUrl,
  });
}
