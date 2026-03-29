FROM nginx:1.25-alpine

COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD wget -qO- http://localhost:80/ || exit 1

EXPOSE 80
