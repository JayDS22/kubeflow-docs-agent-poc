"""
Unit tests for the ingestion pipeline components: chunker and content cleaner.
"""
from ingestion.chunker import clean_content, build_citation_url


class TestContentCleaner:
    def test_strips_yaml_frontmatter(self):
        text = "---\ntitle: Test\nweight: 1\n---\n\nActual content here."
        result = clean_content(text)
        assert "title:" not in result
        assert "Actual content here" in result

    def test_strips_toml_frontmatter(self):
        text = "+++\ntitle = 'Test'\n+++\n\nBody text."
        result = clean_content(text)
        assert "title =" not in result
        assert "Body text" in result

    def test_strips_hugo_shortcodes(self):
        text = "Some text {{< alert >}}warning{{< /alert >}} more text."
        result = clean_content(text)
        assert "{{<" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_strips_html_tags(self):
        text = "<div class='note'><p>Important info</p></div>"
        result = clean_content(text)
        assert "<div" not in result
        assert "Important info" in result

    def test_strips_html_entities(self):
        text = "Use &amp; for ampersand and &lt; for less than."
        result = clean_content(text)
        assert "&amp;" not in result

    def test_collapses_blank_lines(self):
        text = "Line one.\n\n\n\n\nLine two."
        result = clean_content(text)
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert clean_content("") == ""
        assert clean_content("   ") == ""


class TestCitationUrlBuilder:
    def test_standard_doc_path(self):
        url = build_citation_url("content/en/docs/started/installing.md")
        assert url == "https://www.kubeflow.org/docs/started/installing/"

    def test_index_page(self):
        url = build_citation_url("content/en/docs/components/_index.md")
        assert url == "https://www.kubeflow.org/docs/components/"

    def test_html_extension(self):
        url = build_citation_url("content/en/docs/pipelines/overview.html")
        assert url == "https://www.kubeflow.org/docs/pipelines/overview/"

    def test_nested_path(self):
        url = build_citation_url("content/en/docs/components/pipelines/sdk/build-component.md")
        assert url == "https://www.kubeflow.org/docs/components/pipelines/sdk/build-component/"
