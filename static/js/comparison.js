// Progressive enhancement: labeled image pairs remain visible without JavaScript.
// Native range inputs provide pointer, touch, and keyboard interaction.
document.querySelectorAll(".image-comparison").forEach((comparison) => {
  const slider = comparison.querySelector(".comparison-range");
  if (!slider) return;

  const update = () => {
    const position = Number(slider.value);
    comparison.style.setProperty("--reveal", `${position}%`);
    slider.setAttribute("aria-valuetext", `${position}% ground truth, ${100 - position}% ParticleSplat`);
  };

  slider.addEventListener("input", update);
  // Restore the initial position after a back/forward cache navigation as well.
  window.addEventListener("pageshow", update);
  update();
  comparison.classList.add("is-interactive");
  slider.hidden = false;
});
