import { createClient } from '@supabase/supabase-js'

const SB_URL = 'https://sfqlshdlqwqlkxdrfdke.supabase.co'
const SB_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNmcWxzaGRscXdxbGt4ZHJmZGtlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5MDM0NzUsImV4cCI6MjA4MTQ3OTQ3NX0.CMbJ_5IUxAifoNIzqdxu_3sz31AtOMw2vRBPxfxZzSk'

const supabase = createClient(SB_URL, SB_KEY)

export default ({ app }) => {
    try {
        app.config.globalProperties.$supabase = supabase
        console.log('✅ Supabase initialized')
    } catch (e) {
        console.error('❌ Supabase boot failed:', e)
    }
}

export { supabase }
