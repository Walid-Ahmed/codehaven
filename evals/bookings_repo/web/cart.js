const TAX = 0.1;

export function applyCoupon(amount, coupon) {
  return coupon === "SAVE10" ? amount * 0.9 : amount;
}

export function totalPrice(items, coupon) {
  const subtotal = items.reduce((sum, item) => sum + item.price, 0);
  return applyCoupon(subtotal, coupon) * (1 + TAX);
}
